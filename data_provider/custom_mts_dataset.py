import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomMTSDataset(Dataset):
    def __init__(self, root_path, data_path='NP.csv', split='train',
                 context_length=168, prediction_length=24, stride=1,
                 scale=True, features='M', target='-1'):
        super().__init__()
        self.root_path = root_path
        self.data_path = data_path
        self.split = split
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.scale = scale
        self.features = features
        self.target = target
        self.__read_data__()

    def _get_target_index(self, df_data):
        if self.target.lstrip('-').isdigit():
            target_idx = int(self.target)
            num_cols = len(df_data.columns)
            if target_idx < 0:
                target_idx = num_cols + target_idx
                
            if 0 <= target_idx < num_cols:
                col_name = df_data.columns[target_idx]
                logger.info(f"Using target variable by index: {self.target} -> {target_idx} -> '{col_name}'")
                return target_idx
            else:
                raise ValueError(f"Target index {self.target} is out of range. Available indices: 0-{num_cols-1} or -{num_cols}-(-1)")
        else:
           
            try:
                target_idx = df_data.columns.get_loc(self.target)
                logger.info(f"Using target variable by name: '{self.target}' -> index {target_idx}")
                return target_idx
            except KeyError:
                available_cols = list(df_data.columns)
                raise KeyError(f"Target column '{self.target}' not found in the data columns: {available_cols}")

    def __read_data__(self):
        self.csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(self.csv_path)
        data_name = self.data_path.split('.')[0].split('/')[0]
        if 'ETT' in data_name:

            if 'ETTh' in data_name:
               
                train_end = 12*30*24
                val_end = 12*30*24 + 4*30*24
                test_end = 12*30*24 + 8*30*24
            else:
               
                train_end = 12*30*24*4
                val_end = 12*30*24*4 + 4*30*24*4
                test_end = 12*30*24*4 + 8*30*24*4

            border1s = {
                'train': 0, 
                'val': train_end - self.context_length, 
                'test': val_end - self.context_length
            }
            border2s = {
                'train': train_end, 
                'val': val_end, 
                'test': test_end
            }
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2) 
            num_val = len(df_raw) - num_train - num_test
            
            border1s = {
                'train': 0, 
                'val': num_train - self.context_length, 
                'test': len(df_raw) - num_test - self.context_length
            }
            border2s = {
                'train': num_train, 
                'val': num_train + num_val, 
                'test': len(df_raw)
            }
        
        border1 = border1s[self.split]
        border2 = border2s[self.split]

        all_cols = df_raw.columns.drop('date')
        df_data = df_raw[all_cols]
        self.num_variables = df_data.shape[1]

        if self.scale:
            train_data = df_data.iloc[border1s['train']:border2s['train']].values
            self.scaler = StandardScaler()
            self.scaler.fit(train_data)
            self.data = self.scaler.transform(df_data.values)
        else:
            self.data = df_data.values

        self.data_split = self.data[border1:border2]

        self.samples = []
        total_time = len(self.data_split)
        max_start_idx = total_time - (self.context_length + self.prediction_length)
        
        if max_start_idx < 0:
            logger.warning(f"Not enough data in split '{self.split}' to create any samples.")
            return

        if self.features == 'M':
            
            vars_to_iterate = range(self.num_variables)
        elif self.features in ['S', 'MS']:
           
            target_idx = self._get_target_index(df_data)
            vars_to_iterate = [target_idx]
        
        for var_idx in vars_to_iterate:
            for start_idx in range(0, max_start_idx + 1, self.stride):
                self.samples.append((var_idx, start_idx))
        
        logger.info(f"Created {len(self.samples)} samples for split '{self.split}' on '{self.data_path}' (mode: {self.features}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        var_idx, start_idx = self.samples[idx]
        end_ctx = start_idx + self.context_length
        end_pred = end_ctx + self.prediction_length
        labels = self.data_split[end_ctx:end_pred, var_idx]
        
        if self.features == 'MS':
            history = self.data_split[start_idx:end_ctx, :]  
            item = {
                'inputs': torch.tensor(history, dtype=torch.float32),  
                'labels': torch.tensor(labels, dtype=torch.float32),
                'variable_idx': torch.tensor(var_idx, dtype=torch.long)
            }

            item['all_variables_history'] = torch.tensor(history.T, dtype=torch.float32)
            
        else:
            history = self.data_split[start_idx:end_ctx, var_idx]
            
            item = {
                'inputs': torch.tensor(history, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.float32),
                'variable_idx': torch.tensor(var_idx, dtype=torch.long)
            }
            all_vars_history = self.data_split[start_idx:end_ctx, :].T 
            item['all_variables_history'] = torch.tensor(all_vars_history, dtype=torch.float32)
        
        return item