"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

from timm.models import create_model
import argparse
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from sklearn.cluster import KMeans
import open3d as o3d
from M3DM.cpu_knn import fill_missing_values
from feature_extractors.ransac_position import get_registration_np,get_registration_refine_np
from utils.utils import get_args_point_mae
from M3DM.models import Model1
from sklearn.decomposition import PCA


LOGGER = logging.getLogger(__name__)



class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        basic_template=None,
        **kwargs,
    ):
        # self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})
        self.voxel_size = 0.5 #0.1

        # feature_aggregator = patchcore.common.NetworkFeatureAggregator(
        #     self.backbone, self.layers_to_extract_from, self.device
        # )
        # feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        # self.forward_modules["feature_aggregator"] = feature_aggregator

        # preprocessing = patchcore.common.Preprocessing(
        #     feature_dimensions, pretrain_embed_dimension
        # )
        # self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        # self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
        #     device=self.device, target_size=input_shape[-2:]
        # )

        self.featuresampler = featuresampler
        self.dataloader_count = 0
        self.basic_template = basic_template
        self.deep_feature_extractor = None
        # self.pca = PCA(n_components=10)
        
    def set_deep_feature_extractor(self):
        # args = get_args_point_mae()
        self.deep_feature_extractor = Model1(device='cuda', 
                        rgb_backbone_name='vit_base_patch8_224_dino', 
                        xyz_backbone_name='Point_MAE', 
                        group_size = 128, 
                        num_group = 16384)
        self.deep_feature_extractor = self.deep_feature_extractor.cuda()
    
    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
    
    def embed_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed_xyz(data)
    
    def _embed_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        reg_data = reg_data.astype(np.float32)
        return reg_data
    
    def _embed_fpfh(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        return fpfh
    
    def _embed_pointmae(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        return pmae_features,center_idx

    def _embed_downpointmae_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        pointcloud_data = torch.from_numpy(reg_data).permute(1,0).unsqueeze(0).cuda().float()
        pmae_features, center, ori_idx, center_idx = self.deep_feature_extractor(pointcloud_data)
        pmae_features = pmae_features.squeeze(0).permute(1,0).cpu().numpy()
        pmae_features = pmae_features.astype(np.float32)
        # pmae_features = self.pca.fit_transform(pmae_features)
        mask_idx = center_idx.squeeze().long()
        xyz = reg_data[mask_idx.cpu().numpy(),:]
        xyz = xyz.repeat(333,1)
        features = np.concatenate([pmae_features,xyz],axis=1)
        return features.astype(np.float32),center_idx
    
    def _embed_upfpfh_xyz(self, point_cloud, detach=True):
        reg_data = get_registration_np(point_cloud.squeeze(0).cpu().numpy(),self.basic_template)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reg_data))
        radius_normal = self.voxel_size * 2
        o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
        (radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T
        # fpfh = torch.from_numpy(pcd_fpfh.data.T)
        # print(fpfh.shape)
        fpfh = fpfh.astype(np.float32)
        xyz = reg_data.repeat(11,1)
        features = np.concatenate([fpfh,xyz],axis=1)
        return features.astype(np.float32)
    
    # def _embed(self, images, detach=True, provide_patch_shapes=False):
    #     """Returns feature embeddings for images."""

    #     def _detach(features):
    #         if detach:
    #             return [x.detach().cpu().numpy() for x in features]
    #         return features

    #     _ = self.model.eval()
    #     with torch.no_grad():
    #         features = self.model(images)['seg_feat']
    #         features = features.reshape(-1,768)
    #         # print(features.shape)

    #     patch_shapes = [14,14]
    #     if provide_patch_shapes:
    #         return _detach(features), patch_shapes
    #     return _detach(features)
    

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_with_return_feature(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])
        return features
    
    def get_all_features(self, training_data):
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            training_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        features = np.concatenate(features, axis=0)
        return features
    
    def fit_with_limit_size(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def fit_with_limit_size_fpfh(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_fpfh(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_fpfh_upxyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_fpfh_upxyz(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_fpfh_upxyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                
                return self._embed_upfpfh_xyz(input_pointcloud)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_pmae(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_pmae(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_pmae(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_pointmae(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def fit_with_limit_size_downpmae_xyz(self, training_data, limit_size):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        return self._fill_memory_bank_with_limit_size_downpmae_xyz(training_data, limit_size)
        
    def _fill_memory_bank_with_limit_size_downpmae_xyz(self, input_data, limit_size):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_pointcloud):
            with torch.no_grad():
                pmae_features, sample_idx =self._embed_downpointmae_xyz(input_pointcloud)
                return pmae_features

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                features.append(_image_to_features(input_pointcloud))

        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict(input_pointcloud)
                # for score, mask in zip(_scores, _masks):
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        # images = images.to(torch.float).to(self.device)
        # _ = self.forward_modules.eval()

        # batchsize = images.shape[0]
        with torch.no_grad():
            features = self._embed_xyz(input_pointcloud)
            # print(patch_shapes) [32,32]
            features = np.asarray(features)
            # print(features.shape)
            # features = np.repeat(features,2,axis=1)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            # print(patch_scores.shape)
            # print(image_scores)
            # image_scores = self.patch_maker.unpatch_scores(
            #     image_scores, batchsize=batchsize
            # )
            # image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # print(image_scores.shape)
            # image_scores = self.patch_maker.score(image_scores)
            # print(image_scores.shape)

            # patch_scores = self.patch_maker.unpatch_scores(
            #     patch_scores, batchsize=batchsize
            # )
            # patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [image_scores], [mask for mask in patch_scores]
        # return [score for score in image_scores], [mask for mask in image_scores]
    
    def predict_fpfh(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh(data)
        return self._predict_fpfh(data)

    def _predict_dataloader_fpfh(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fpfh(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_fpfh(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]

    def predict_fpfh_upxyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_fpfh_upxyz(data)
        return self._predict_fpfh_upxyz(data)

    def _predict_dataloader_fpfh_upxyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_fpfh_upxyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_fpfh_upxyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features = self._embed_upfpfh_xyz(input_pointcloud)
            features = np.asarray(features)
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
        return [image_scores], [mask for mask in patch_scores]
    
    def predict_pmae(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_pmae(data)
        return self._predict_pmae(data)

    def _predict_dataloader_pmae(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_pmae(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_pmae(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_pointmae(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]

    
    def predict_downpmae_xyz(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader_downpmae_xyz(data)
        return self._predict_downpmae_xyz(data)

    def _predict_dataloader_downpmae_xyz(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for input_pointcloud, mask, label, path in data_iterator:
                labels_gt.extend(label.numpy().tolist())
                masks_gt.extend(mask.numpy().tolist())
                _scores, _masks = self._predict_downpmae_xyz(input_pointcloud)
                scores.extend(_scores)
                masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict_downpmae_xyz(self, input_pointcloud):
        """Infer score and mask for a batch of images."""
        with torch.no_grad():
            features, sample_dix = self._embed_downpointmae_xyz(input_pointcloud)
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = np.max(image_scores)
            mask_idx = sample_dix.squeeze().long()
            xyz_sampled = input_pointcloud[0][mask_idx.cpu(),:]
            # print(patch_scores.shape)
            # print(input_pointcloud.shape)
            # print(xyz_sampled.shape)
            full_scores = fill_missing_values(xyz_sampled,patch_scores,input_pointcloud[0], k=1)
        return [image_scores], [mask for mask in full_scores]
    
    def _predict_past_tasks(self, features, data):
        pass
            
    def _fit_past_tasks(self, features, data):
        pass
        

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
