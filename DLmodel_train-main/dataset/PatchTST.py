from torch.utils.data import Dataset
import os, glob
import torch
import numpy as np
import json
from collections import defaultdict
import random
import torch.nn.functional as Fu
import pandas as pd
import time

class Dataset_TST_Deadlift(Dataset):
    def __init__(self, dataset_root):
        self.sample_paths = []  
        self.data = {}
        self.features = []
        self.labels = []
        self.dim = int
        
        # 對應資料夾名稱 → label 數字
        category_map = {
            'Category_1': 0,
            'Category_2': 1,
            'Category_3': 2,
            'Category_4': 3,
            'Category_5': 4 
        }

        class_recs = defaultdict(list)

        for cat_name, label in category_map.items():
            path = os.path.join(dataset_root, cat_name)
            if not os.path.isdir(path):
                continue
            recordings = glob.glob(os.path.join(path, '*'))
            class_recs[str(label)].extend(recordings)

        # 類別 0 → 全為 0 的標籤
        for recording in class_recs['0']:
            self.data[os.path.basename(recording)] = [0, 0, 0, 0]
            
        print('0 category have', len(class_recs['0']), 'videos')
        # 其他類別（1~4），建立多標籤
        for label_str, recordings in class_recs.items():
            if label_str == '0':
                continue
            label = int(label_str)
            for recording in recordings:
                if os.path.basename(recording) not in self.data:
                    self.data[os.path.basename(recording)] = [0, 0, 0, 0]
                self.data[os.path.basename(recording)][label - 1] = 1
            print(label_str, 'category have', len(recordings), 'videos')

        # 寫入 JSON
        os.makedirs(dataset_root, exist_ok=True)
        with open(os.path.join(dataset_root, 'label.json'), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)    
        
        for recording, label in list(self.data.items()):
            recording_path = self.find_folder(dataset_root, recording)
            if recording_path == None:
                print('not available', recording)
                continue
            delta_path = os.path.join(recording_path, 'filtered_delta_norm')
            delta2_path = os.path.join(recording_path, 'filtered_delta2_norm')
            square_path = os.path.join(recording_path, 'filtered_delta_square_norm')
            zscore_path = os.path.join(recording_path, 'filtered_zscore_norm')
            orin_path = os.path.join(recording_path, 'filtered_norm')
            
            if not all(map(os.path.exists, [delta_path, delta2_path, zscore_path, square_path, orin_path])):
                print(f"Missing data in {recording_path}")
            
            deltas = glob.glob(os.path.join(delta_path, '*.txt'))
            delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
            squares = glob.glob(os.path.join(square_path, '*.txt'))
            zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
            orins = glob.glob(os.path.join(orin_path, '*.txt'))
            
            data_per_ind = list(self.fetch(zip(deltas, delta2s, zscores, squares, orins)))
            self.features.extend(torch.tensor(data_per_ind).float())
            self.labels.extend([torch.tensor(label).float()] * len(data_per_ind))
        print('total data:', len(self.features))
        print('total label', len(self.labels))
    
    def find_folder(self, root_path, target_folder_name):
        for dirpath, dirnames, filenames in os.walk(root_path):
            if target_folder_name in dirnames:
                return os.path.join(dirpath, target_folder_name)
        return None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
            
        return x, y, idx
    
    def add_gaussian_noise(self, x, std=0.01):
        """
        x: tensor (T, F)
        std: 標準差，決定噪音強度
        """
        noise = torch.randn_like(x) * std
        return x + noise
        
    def time_stretch(self, x, stretch_factor):
        """
        x: tensor (T=110, F)
        stretch_factor: float, >1 表示拉長，<1 表示壓縮
        """
        T, F = x.shape
        new_T = int(T * stretch_factor)

        # 線性插值變更時間長度
        x_stretched = Fu.interpolate(x.T.unsqueeze(0), size=new_T, mode='linear', align_corners=True)
        x_stretched = x_stretched.squeeze(0).T

        # 補回或裁切回原始長度 110
        if new_T < T:
            pad_len = T - new_T
            padding = torch.zeros(pad_len, F, device=x.device)
            x_stretched = torch.cat([x_stretched, padding], dim=0)
        elif new_T > T:
            x_stretched = x_stretched[:T]

        return x_stretched
    
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
                # 將 num 裡的數據 flatten 
                frame_data = [item for sublist in num for item in sublist]
                self.dim = len(frame_data)
                data_per_ind.append(frame_data)
                
                if len(data_per_ind) == 110:  # 达到110帧时返回
                    yield data_per_ind
                    data_per_ind = []
                    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]
    
class Dataset_TST_Benchpress(Dataset):
    def __init__(self, dataset_root):
        self.sample_paths = []  
        self.features = []
        self.labels = []
        
        df = pd.read_csv(dataset_root, skiprows=1)
        count = 0
        tmp_data = []

        for _, row in df.iterrows():
            data_1 = row.iloc[0:35].values.astype(float)
            data_2 = row.iloc[40:65].values.astype(float)
            data_3 = row.iloc[70:75].values.astype(float)
            data = data_1.tolist() + data_2.tolist() + data_3.tolist()
            label = row.iloc[77:81].values.astype(int)
            path = row.iloc[-1]

            tmp_data.append(data)
            self.sample_paths.append(path)

            count += 1
            if count == 100:
                block = np.array(tmp_data)  # shape = (100, 27)
                self.features.append(torch.tensor(block).float())
                self.labels.append(torch.tensor(label).float())

                tmp_data = []
                count = 0

        self.features = torch.stack(self.features)  # ✅ shape: (N, 100, 27)
        self.labels = torch.stack(self.labels)      # ✅ shape: (N, 7)
        self.dim = self.features.shape[-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx
                    
    def get_sample_path(self, idx):
        return self.sample_paths[idx]

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.time_stretch(x, random.uniform(0.8, 1.2))
            x = self.add_gaussian_noise(x, std=0.01)
        return x, y, true_idx

    def time_stretch(self, x, stretch_factor):
        # 假設 x.shape = (T, F)
        T, F = x.shape
        new_T = int(T * stretch_factor)
        x_stretched = torch.nn.functional.interpolate(
            x.unsqueeze(0).permute(0, 2, 1),  # (1, F, T)
            size=new_T,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1).squeeze(0)  # 回到 (T, F)
        if new_T < T:
            pad = torch.zeros(T - new_T, F, dtype=x.dtype, device=x.device)
            x_stretched = torch.cat([x_stretched, pad], dim=0)
        else:
            x_stretched = x_stretched[:T]
        return x_stretched

    def add_gaussian_noise(self, x, std=0.01):
        noise = torch.randn_like(x) * std
        return x + noise