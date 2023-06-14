# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
from M3DM.m3dm_runner import M3DM
from dataset_pc import real3d_classes

import pandas as pd


def run_3d_ads(args):
    classes = real3d_classes()

    METHOD_NAMES = [args.xyz_backbone_name]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    image_ap_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_ap_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    print('Task start: M3DM '+args.xyz_backbone_name)
    for cls in classes:
        # print(cls)
        model = M3DM(args)
        model.fit(cls)
        image_rocaucs, pixel_rocaucs,image_ap, pixel_ap = model.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        image_ap_df[cls.title()] = image_ap_df['Method'].map(image_ap)
        pixel_ap_df[cls.title()] = pixel_ap_df['Method'].map(pixel_ap)

        print('Task:{}, object_auroc:{}, point_auroc:{}, object_aupr:{}, point_aupr:{}'.format
                    (cls,image_rocaucs_df['Method'].map(image_rocaucs),pixel_rocaucs_df['Method'].map(pixel_rocaucs),image_ap_df['Method'].map(image_ap),pixel_ap_df['Method'].map(pixel_ap)))
        # print(f"\nFinished running on class {cls}")
        # print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    image_ap_df['Mean'] = round(image_ap_df.iloc[:, 1:].mean(axis=1),3)
    pixel_ap_df['Mean'] = round(pixel_ap_df.iloc[:, 1:].mean(axis=1),3)

    # print("\n\n################################################################################")
    # print("############################# Object AUROC Results #############################")
    # print("################################################################################\n")
    # print(image_rocaucs_df.to_markdown(index=False))

    # print("\n\n################################################################################")
    # print("############################# Point AUROC Results #############################")
    # print("################################################################################\n")
    # print(pixel_rocaucs_df.to_markdown(index=False))

    # print("\n\n################################################################################")
    # print("############################# Object AUPR Results #############################")
    # print("################################################################################\n")
    # print(image_ap_df.to_markdown(index=False))

    # print("\n\n################################################################################")
    # print("############################# Point AUPR Results #############################")
    # print("################################################################################\n")
    # print(pixel_ap_df.to_markdown(index=False))



    # with open(args.result_md_path+"image_rocauc_results.md", "a") as tf:
    #     tf.write(image_rocaucs_df.to_markdown(index=False))
    # with open(args.result_md_path+"pixel_rocauc_results.md", "a") as tf:
    #     tf.write(pixel_rocaucs_df.to_markdown(index=False))
    # with open(args.result_md_path+"image_ap_results.md", "a") as tf:
    #     tf.write(image_ap_df.to_markdown(index=False))
    # with open(args.result_md_path+"pixel_ap_results.md", "a") as tf:
    #     tf.write(pixel_ap_df.to_markdown(index=False))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--xyz_backbone_name', default='Point_Bert', type=str, choices=['Point_MAE', 'Point_Bert'],
                        help='Checkpoints name of RGB backbone[Point_MAE, Point_Bert].')
    parser.add_argument('--save_checkpoint_path', default = "./checkpoints/Point-BERT.pth", type=str,
                        choices=["./checkpoints/pointmae_pretrain.pth", "./checkpoints/Point-BERT.pth"],
                        help='Save feature for training fusion block.')

    parser.add_argument('--save_preds', default=False, action='store_true',
                        help='Save predicts results.')
    parser.add_argument('--group_size', default=128, type=int,
                        help='Point group size of Point Transformer.')
    parser.add_argument('--num_group', default=16384, type=int,
                        help='Point groups number of Point Transformer.')
    parser.add_argument('--random_state', default=None, type=int,
                        help='random_state for random project')
    parser.add_argument('--dataset_path', default='./data', type=str, 
                        help='Dataset store path')
    parser.add_argument('--save_path_full', default='./benchmark/point_bert/3dad_m3dm_visualization_full/', type=str, 
                        help='Dataset store path')
    parser.add_argument('--save_path', default='./benchmark/point_bert/3dad_m3dm_visualization/', type=str, 
                        help='Dataset store path')
    parser.add_argument('--result_md_path', default='results/', type=str, 
                        help='Dataset store path')

    parser.add_argument('--max_sample', default=400, type=int,
                        help='Max sample number.')
    parser.add_argument('--checkpoint_path', default='./data', type=str, 
                        help='Dataset store path')


    parser.add_argument('--img_size', default=224, type=int,
                        help='Images size for model')
    parser.add_argument('--coreset_eps', default=0.9, type=float,
                        help='eps for sparse project')
    parser.add_argument('--f_coreset', default=0.1, type=float,
                        help='eps for sparse project')
    parser.add_argument('--rm_zero_for_project', default=False, action='store_true',
                        help='Save predicts results.')
    
    args = parser.parse_args()
    run_3d_ads(args)

# python main.py --xyz_backbone_name Point_MAE --save_checkpoint_path ./checkpoints/pointmae_pretrain.pth
# python main.py --xyz_backbone_name Point_Bert --save_checkpoint_path ./checkpoints/Point-BERT.pth
