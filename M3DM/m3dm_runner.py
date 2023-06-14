import torch
from tqdm import tqdm
import os
from M3DM import multiple_features

from torch.utils.data import DataLoader
from dataset_m3dm import Dataset3dad_train,Dataset3dad_test
# from thop import profile


class M3DM():
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        self.methods = {
                "Point_MAE": multiple_features.PointFeatures(args),
            }
        self.root_dir = args.dataset_path
        self.save_path_full = args.save_path_full
        self.save_path = args.save_path
        if(not os.path.exists(self.save_path_full)):
            os.makedirs(self.save_path_full)
        if(not os.path.exists(self.save_path)):
            os.makedirs(self.save_path)


    def fit(self, class_name):
        train_loader = DataLoader(Dataset3dad_train(self.root_dir, class_name, 2048, True), num_workers=1,batch_size=1, shuffle=True, drop_last=True)

        flag = 0
        for sample,_,_,_ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            # print(flag)
            for method in self.methods.values():
                method.add_sample_to_mem_bank(sample)
                flag += 1
            if flag > self.count:
                flag = 0
                break
                
        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset(class_name)
            

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        image_ap = dict()
        pixel_ap = dict()
        test_loader = DataLoader(Dataset3dad_test(self.root_dir, class_name, 2048, True), num_workers=1,batch_size=1, shuffle=True, drop_last=False)
        path_list = []
        with torch.no_grad():
        
            for sample, mask, label, pcd_path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, mask, label,pcd_path,self.save_path_full,self.save_path)
                    path_list.append(pcd_path)
                        

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            image_rocaucs[method_name] = round(method.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
            image_ap[method_name] = round(method.image_ap, 3)
            pixel_ap[method_name] = round(method.pixel_ap, 3)
            print(f'Class: {class_name}, {method_name} Object AUROC: {method.image_rocauc:.3f}, {method_name} Point AUROC: {method.pixel_rocauc:.3f}, {method_name} Object AUPR: {method.image_ap:.3f}, {method_name} Point AuPR: {method.pixel_ap:.3f}')
            if self.args.save_preds:
                method.save_prediction_maps('./pred_maps', path_list)
        return image_rocaucs, pixel_rocaucs,image_ap, pixel_ap
