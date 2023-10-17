import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.utils
import patchcore.sampler
import patchcore.metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import label
from bisect import bisect
import time
from dataset_pc import Dataset3dad_train,Dataset3dad_test
from torch.utils.data import DataLoader
import open3d as o3d
from utils.visualization import save_anomalymap

import argparse

LOGGER = logging.getLogger(__name__)


@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
# Parameters for Glue-code (to merge different parts of the pipeline.
# @click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
# @click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
# @click.option("--patchsize", type=int, default=3)
# @click.option("--patchscore", type=str, default="max")
# @click.option("--patchoverlap", type=float, default=0.0)
# @click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    gpu,
    seed,
    memory_size,
    anomaly_scorer_num_nn,
    faiss_on_gpu,
    faiss_num_workers
):
    methods = {key: item for (key, item) in methods}

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    root_dir = './data'
    save_root_dir = './benchmark/patchcore_pointmae/'
    print('Task start: Patchcore PointMAE')
    
    real_3d_classes = ['airplane','car','candybar','chicken',
                   'diamond','duck','fish','gemstone',
                   'seahorse','shell','starfish','toffees']

    for dataset_count, dataset_name in enumerate(real_3d_classes):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataset_name,
                dataset_count + 1,
                len(real_3d_classes),
            )
        )
        if( not os.path.exists(save_root_dir+dataset_name)):
            os.makedirs(save_root_dir+dataset_name)
        patchcore.utils.fix_seeds(seed, device)
        train_loader = DataLoader(Dataset3dad_train(root_dir, dataset_name, 1024, True), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=False)
        test_loader = DataLoader(Dataset3dad_test(root_dir, dataset_name, 1024, True), num_workers=1,
                                batch_size=1, shuffle=False, drop_last=False)

        for data, mask, label, path in train_loader:
            basic_template = data.squeeze(0).cpu().numpy()
            break
        
        with device_context:
            torch.cuda.empty_cache()
            sampler = methods["get_sampler"](
                device,
            )
            # PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            PatchCore = patchcore.patchcore.PatchCore(device)
            PatchCore.load(
                backbone=None,
                layers_to_extract_from=None,
                device=device,
                input_shape=None,
                pretrain_embed_dimension=1024,
                target_embed_dimension=1024,
                patchsize=16,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
                basic_template=basic_template,
            )
            
            # point mae
            PatchCore.set_deep_feature_extractor()
            torch.cuda.empty_cache()
            memory_feature = PatchCore.fit_with_limit_size_pmae(train_loader, memory_size)
            aggregator_fpfh = {"scores": [], "segmentations": []}
            start_time = time.time()
            scores_fpfh, segmentations_fpfh, labels_gt, masks_gt = PatchCore.predict_pmae(
                test_loader
            )
            aggregator_fpfh["scores"].append(scores_fpfh)
            # aggregator["segmentations"].append(segmentations)
            scores_fpfh = np.array(aggregator_fpfh["scores"])
            min_scores_fpfh = scores_fpfh.min(axis=-1).reshape(-1, 1)
            max_scores_fpfh = scores_fpfh.max(axis=-1).reshape(-1, 1)
            scores_fpfh = (scores_fpfh - min_scores_fpfh) / (max_scores_fpfh - min_scores_fpfh)
            scores_fpfh = np.mean(scores_fpfh, axis=0)
            ap_seg_fpfh = np.asarray(segmentations_fpfh)
            ap_seg_fpfh = ap_seg_fpfh.flatten()
            min_seg_fpfh = np.min(ap_seg_fpfh)
            max_seg_fpfh = np.max(ap_seg_fpfh)
            ap_seg_fpfh = (ap_seg_fpfh-min_seg_fpfh)/(max_seg_fpfh-min_seg_fpfh)
            end_time = time.time()
            time_cost = (end_time - start_time)/len(test_loader)

            LOGGER.info("Computing evaluation metrics.")
            scores = scores_fpfh
            ap_seg = ap_seg_fpfh
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt
            )["auroc"]
            img_ap = average_precision_score(labels_gt,scores)
            ap_mask = np.concatenate(np.asarray(masks_gt))
            ap_mask = ap_mask.flatten().astype(np.int32)
            pixel_ap = average_precision_score(ap_mask,ap_seg)
            full_pixel_auroc = roc_auc_score(ap_mask,ap_seg)
            print('Task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}, time_cost:{}'.format
                    (dataset_name,auroc,full_pixel_auroc,img_ap,pixel_ap,time_cost))
            # cur_pc_idx = 0
            # for pointcloud, mask, label, sample_path in test_loader:
            #     pc_length = pointcloud.shape[1]
            #     anomaly_cur = ap_seg[cur_pc_idx:cur_pc_idx+pc_length]
            #     path_list = sample_path[0].split('/')
            #     save_anomalymap(sample_path[0],anomaly_cur,os.path.join(save_root_dir,dataset_name,path_list[-1]))
            #     cur_pc_idx = cur_pc_idx+pc_length

    # # Store all results and mean scores to a csv-file.
    # result_metric_names = list(result_collect[-1].keys())[1:]
    # result_dataset_names = [results["dataset_name"] for results in result_collect]
    # result_scores = [list(results.values())[1:] for results in result_collect]
    # patchcore.utils.compute_and_store_final_results(
    #     run_save_path,
    #     result_scores,
    #     column_names=result_metric_names,
    #     row_names=result_dataset_names,
    # )



@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    # config = parser.parse_known_args()[-1][0]
    # subparser = parser.add_subparsers(dest='subparser_name')

    # from patchcore.configs.mvtecad_dualprompt import get_args_parser
    # config_parser = subparser.add_parser('mvtecad_dualprompt', help='MVTec AD')
    # get_args_parser(config_parser)
    # args = parser.parse_args()
    # print(args)

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
