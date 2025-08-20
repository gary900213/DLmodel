from torch.utils.data import Dataset
import os, glob
import torch
import numpy as np
import json
from collections import defaultdict
import random
import torch.nn.functional as Fu

class Dataset_dd2voz(Dataset):
    def __init__(self, dataset, GT_class):
        self.sample_paths = []   
        self.features = []       
        self.labels = []
        self.count_info = []    
        self.dim = int
        counter = 0   
        
        self.missing = []  
        
        valid_categories = {'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5'}
        categories = [path for path in glob.glob(os.path.join(dataset, '*')) if os.path.basename(path) in valid_categories]
        
        T_recordings = os.listdir(os.path.join(dataset, f'Category_{GT_class}'))
        
        for category in categories:
            recordings = glob.glob(os.path.join(category, '*'))
            for recording in recordings:
                # 判斷此影片是否在正確的分類
                if os.path.basename(category) != f'Category_{GT_class}':
                    if os.path.basename(recording) in T_recordings:
                        print(f'{recording} is overlapped with True recording.')
                        continue
                        
                delta_path = os.path.join(recording, 'filtered_delta_norm')
                delta2_path = os.path.join(recording, 'filtered_delta2_norm')
                square_path = os.path.join(recording, 'filtered_delta_square_norm')
                zscore_path = os.path.join(recording, 'filtered_zscore_norm')
                orin_path = os.path.join(recording, 'filtered_norm')
                
                if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                    print(f"Missing data in {recording}")
                    continue
                
                deltas = glob.glob(os.path.join(delta_path, '*.txt'))
                delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
                squares = glob.glob(os.path.join(square_path, '*.txt'))
                zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
                orins = glob.glob(os.path.join(orin_path, '*.txt'))
                
                data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
                self.features.extend(data_per_ind)
                self.labels.extend([1 if f'Category_{GT_class}' in category else 0] * len(data_per_ind))
                for _ in range(len(data_per_ind)):
                    counter += 1
                
            print(f'{category} have {counter} samples')
            self.count_info.append(counter)
            counter = 0
            
        with open("missing_merge.txt", "w", encoding="utf-8") as f:
            for line in self.missing:
                f.write(line + "\n")
    
    def get_ratio(self):
        category_ratio = {}
        total = sum(self.count_info)
        ratios = [x / total for x in self.count_info]
        for i, ratio in enumerate(ratios):
            category_ratio[f'{i+1}'] = ratio
        return category_ratio
    
    def fetch(self, uds):
        """
        处理数据并返回特征和对应的文件路径
        """
        data_per_ind = []
        
        # 對每一下做處理
        for ud in uds:
            parsed_data = []
            
            for file in ud:
                with open(file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    parsed_data.append([list(map(float, line.split(','))) for line in lines])
            self.sample_paths.append(file)
            
            for num in zip(*parsed_data):
                # 將 num 裡的數據做 flat 
                frame_data = [item for sublist in num for item in sublist]
                self.dim = len(frame_data)
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []
            if len(frame_data) != 30:
                self.missing.append(ud[0])
                
        


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.features[idx], dtype=torch.float32),
        torch.tensor(self.labels[idx], dtype=torch.long),
        idx  # 把原始 full_dataset 的 index 傳出
    )
    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]
    
class Dataset_SHAP(Dataset):
    def __init__(self, dataset, GT_class, mode):
        self.mode = mode
        self.sample_paths = []   
        self.features = []       
        self.labels = []    
        self.count_info = []
        self.dim = int
        counter = 0     
        
        valid_categories = {'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5'}
        categories = [path for path in glob.glob(os.path.join(dataset, '*')) if os.path.basename(path) in valid_categories]
        
        T_recordings = os.listdir(os.path.join(dataset, f'Category_{GT_class}'))
        
        for category in categories:
            recordings = glob.glob(os.path.join(category, '*'))
            for recording in recordings:
                # 判斷此影片是否在正確的分類
                if os.path.basename(category) != f'Category_{GT_class}':
                    if os.path.basename(recording) in T_recordings:
                        print(f'{recording} is overlapped with True recording.')
                        continue
                        
                delta_path = os.path.join(recording, 'filtered_delta_norm')
                delta2_path = os.path.join(recording, 'filtered_delta2_norm')
                square_path = os.path.join(recording, 'filtered_delta_square_norm')
                zscore_path = os.path.join(recording, 'filtered_zscore_norm')
                orin_path = os.path.join(recording, 'filtered_norm')
                
                if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                    print(f"Missing data in {recording}")
                    continue
                
                deltas = glob.glob(os.path.join(delta_path, '*.txt'))
                delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
                squares = glob.glob(os.path.join(square_path, '*.txt'))
                zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
                orins = glob.glob(os.path.join(orin_path, '*.txt'))
                
                data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
                self.features.extend(data_per_ind)
                self.labels.extend([1 if f'Category_{GT_class}' in category else 0] * len(data_per_ind))
                for _ in range(len(data_per_ind)):
                    counter += 1
                
            print(f'{category} have {counter} samples')
            self.count_info.append(counter)
            counter = 0
    
    def get_ratio(self):
        category_ratio = {}
        total = sum(self.count_info)
        ratios = [x / total for x in self.count_info]
        for i, ratio in enumerate(ratios):
            category_ratio[f'{i+1}'] = ratio
        return category_ratio
    
    def fetch(self, uds):
        """
        处理数据并返回特征和对应的文件路径
        """
        data_per_ind = []
        
        # 對每一下做處理
        for ud in uds:
            parsed_data = []
            
            # SHAP_abs
            for file in ud:
                process_type = os.path.basename(os.path.dirname(file))
                skip_idx = self.skip_det(process_type)
                    
                with open(file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    parsed_data.append([
                        [float(x) for i, x in enumerate(line.split(',')) if i not in skip_idx] # 0:膝角, 1:髖角, 2:身體長度, 3:bar_x, 4:bar_y
                        for line in lines
                    ])
            self.sample_paths.append(file)
            
            for num in zip(*parsed_data):
                # 將 num 裡的數據變成 19*1 
                frame_data = [item for sublist in num for item in sublist]
                self.dim = len(frame_data)
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []
    
    def skip_det(self, type):
        if self.mode == 'abs':
            if type == 'filtered_delta_square_norm':
                skip_idx = [2, 3] # delta_square : body_length, bar_x
            elif type == 'filtered_zscore_norm':
                skip_idx = [] # zscore : 
            elif type == 'filtered_norm':
                skip_idx = [3, 4] # filtered : bar_x, bar_y
            elif type == 'filtered_delta2_norm':
                skip_idx = [3] # delta2 : bar_x
            elif type == 'filtered_delta_norm':
                skip_idx = [] # delta : 
            else:
                skip_idx = []
            
        if self.mode == 'avg_abs_min':
            if type == 'filtered_delta_square_norm':
                skip_idx = [3] # delta_square : bar_x
            elif type == 'filtered_zscore_norm':
                skip_idx = [] # zscore : 
            elif type == 'filtered_norm':
                skip_idx = [] # filtered : 
            elif type == 'filtered_delta2_norm':
                skip_idx = [] # delta2 : 
            elif type == 'filtered_delta_norm':
                skip_idx = [0, 3] # delta : bar_x, knee_angle
            else:
                skip_idx = []
                
        if self.mode == 'avg_min':
            if type == 'filtered_delta_square_norm':
                skip_idx = [] # delta_square : 
            elif type == 'filtered_zscore_norm':
                skip_idx = [0, 1] # zscore : knee_angle, hip_angle
            elif type == 'filtered_norm':
                skip_idx = [0, 1] # filtered : knee_angle, hip_angle
            elif type == 'filtered_delta2_norm':
                skip_idx = [] # delta2 : 
            elif type == 'filtered_delta_norm':
                skip_idx = [4] # delta : bar_y
            else:
                skip_idx = []
        return skip_idx

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.features[idx], dtype=torch.float32),
        torch.tensor(self.labels[idx], dtype=torch.long),
        idx  # 把原始 full_dataset 的 index 傳出
    )
    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]
    
class Dataset_3D(Dataset):
    def __init__(self, dataset, GT_class):
        self.sample_paths = []   
        self.features = []       
        self.labels = []
        self.count_info = []    
        self.dim = int
        counter = 0     
        
        valid_categories = {'Category_1', 'Category_2', 'Category_3', 'Category_4', 'Category_5'}
        categories = [path for path in glob.glob(os.path.join(dataset, '*')) if os.path.basename(path) in valid_categories]
        
        T_recordings = os.listdir(os.path.join(dataset, f'Category_{GT_class}'))
        
        for category in categories:
            recordings = glob.glob(os.path.join(category, '*'))
            for recording in recordings:
                # 判斷此影片是否在正確的分類
                if os.path.basename(category) != f'Category_{GT_class}':
                    if os.path.basename(recording) in T_recordings:
                        print(f'{recording} is overlapped with True recording.')
                        continue
                        
                delta_path = os.path.join(recording, 'filtered_delta_norm')
                delta2_path = os.path.join(recording, 'filtered_delta2_norm')
                square_path = os.path.join(recording, 'filtered_delta_square_norm')
                zscore_path = os.path.join(recording, 'filtered_zscore_norm')
                orin_path = os.path.join(recording, 'filtered_norm')
                
                if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                    print(f"Missing data in {recording}")
                    continue
                
                deltas = glob.glob(os.path.join(delta_path, '*.txt'))
                delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
                squares = glob.glob(os.path.join(square_path, '*.txt'))
                zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
                orins = glob.glob(os.path.join(orin_path, '*.txt'))
                
                data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
                self.features.extend(data_per_ind)
                self.labels.extend([1 if f'Category_{GT_class}' in category else 0] * len(data_per_ind))
                for _ in range(len(data_per_ind)):
                    counter += 1
                
            print(f'{category} have {counter} samples')
            self.count_info.append(counter)
            counter = 0
    
    def get_ratio(self):
        category_ratio = {}
        total = sum(self.count_info)
        ratios = [x / total for x in self.count_info]
        for i, ratio in enumerate(ratios):
            category_ratio[f'{i+1}'] = ratio
        return category_ratio
    
    def fetch(self, uds):
        """
        处理数据并返回特征和对应的文件路径
        """
        data_per_ind = []
        
        # 對每一下做處理
        for ud in uds:
            parsed_data = []
            
            for file in ud:
                with open(file, 'r') as f:
                    lines = f.read().strip().split('\n')
                    parsed_data.append([list(map(float, line.split(','))) for line in lines])
            self.sample_paths.append(file)
            
            for num in zip(*parsed_data):
                # 將 num 裡的數據變成 25*1 
                frame_data = [item for sublist in num for item in sublist]
                self.dim = len(frame_data)
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
        torch.tensor(self.features[idx], dtype=torch.float32),
        torch.tensor(self.labels[idx], dtype=torch.long),
        idx  # 把原始 full_dataset 的 index 傳出
    )
    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]