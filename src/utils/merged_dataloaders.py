"""
__version__: 0.0

This file provides wrapper functions to create DataLoaders that combine
data from multiple sources (survey, diary, sensor) for a single user.

The core challenge is aligning data (ensuring 'pm96' survey data is
batched with 'pm96' sequence data) and handling heterogeneous data
(static features + variable-length sequences).

This is solved by:
1.  A `MergedDataset` that loads all data in its __init__ and builds
    dictionaries (e.g., `user_id -> features`).
2.  This dataset's "index" is a list of user_ids that are present
    in ALL requested data sources (the intersection).
3.  A `collate_fn_merged` that receives a list of dictionary samples
    and correctly stacks the static data and pads the sequence data.

Outputs from the wrappers are DataLoaders that yield a single batch 
dictionary. For example, for get_merged_all_loader:

batch = {
    'survey_features': (Tensor: B, F_survey),
    'survey_labels':   (Tensor: B, F_labels),
    'diary_features':  (Tensor: B, L_diary_max, F_diary),
    'diary_lengths':   (Tensor: B,),
    'sensor_features': (Tensor: B, L_sensor_max, F_sensor),
    'sensor_lengths':  (Tensor: B,),  <-- (Only if sensor_mode='user')
    'user_ids':        (list[str]: B)
}

refer to test.test_merged_dataloader.py to learn how to use the wrappers.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
try:
    from src.utils.base_dataloaders import SurveyDataset, SequenceDataset, SensorHRVDataset, BucketBatchSampler
except ImportError:
    from base_dataloaders import SurveyDataset, SequenceDataset, SensorHRVDataset


class MergedDataset(Dataset):
    """
    A Dataset that merges static survey data with one or more
    sequence datasets (diary, sensor) for a given user.
    
    It finds the intersection of users present in all requested
    datasets to ensure every sample is complete.
    """
    def __init__(self,
                 survey_csv=None, diary_csv=None, sensor_csv=None,
                 survey_args=None, diary_args=None, sensor_args=None,
                 use_diary=False, use_sensor=False):
        
        self.use_diary = use_diary
        self.use_sensor = use_sensor
        self.sensor_mode = sensor_args.get('mode', 'user') if sensor_args else 'user'
        
        # --- 1. Load Survey Data (Base) ---
        # This is required and forms the base of our dataset
        if survey_csv is None or survey_args is None:
            raise ValueError("survey_csv and survey_args must be provided")
            
        survey_dataset = SurveyDataset(csv_file=survey_csv, **survey_args)
        
        self.survey_features_dict = {}
        self.survey_labels_dict = {}
        
        # Build dictionaries from the SurveyDataset
        for features, label, user_id in survey_dataset:
            self.survey_features_dict[user_id] = features
            self.survey_labels_dict[user_id] = label
            
        # Initialize the user list from survey data
        user_id_set = set(self.survey_features_dict.keys())

        # --- 2. (Conditional) Load Sleep Diary Data ---
        self.diary_features_dict = {}
        self.diary_lengths_dict = {}
        if self.use_diary:
            if diary_csv is None or diary_args is None:
                raise ValueError("diary_csv and diary_args must be provided")
                
            diary_dataset = SequenceDataset(csv_file=diary_csv, **diary_args)
            for features, user_id in diary_dataset:
                self.diary_features_dict[user_id] = features
                self.diary_lengths_dict[user_id] = len(features)
            
            # Find the intersection of users
            diary_users = set(self.diary_features_dict.keys())
            user_id_set.intersection_update(diary_users)

        # --- 3. (Conditional) Load Sensor HRV Data ---
        self.sensor_features_dict = {}
        self.sensor_lengths_dict = {}
        if self.use_sensor:
            if sensor_csv is None or sensor_args is None:
                raise ValueError("sensor_csv and sensor_args must be provided")
            
            # Make a copy to avoid modifying the original dict
            sensor_args_copy = sensor_args.copy()
            
            # Pop 'batch_size' if it exists. It's an arg for the 
            # DataLoader wrapper, not the Dataset constructor.
            sensor_args_copy.pop('batch_size', None)
            
            sensor_dataset = SensorHRVDataset(csv_file=sensor_csv, **sensor_args_copy)
            
            for features, user_id in sensor_dataset:
                self.sensor_features_dict[user_id] = features
                if self.sensor_mode == 'user':
                    self.sensor_lengths_dict[user_id] = len(features)
            
            # Find the intersection
            sensor_users = set(self.sensor_features_dict.keys())
            user_id_set.intersection_update(sensor_users)

        # --- 4. Finalize User List ---
        # self.final_user_ids contains only users present in ALL requested datasets
        self.final_user_ids = sorted(list(user_id_set))
        
        self.diary_lengths = []
        self.sensor_lengths = []
        
        # In-order lengths for the BucketBatchSampler
        for user_id in self.final_user_ids:
            if self.use_diary:
                self.diary_lengths.append(self.diary_lengths_dict[user_id])
            if self.use_sensor and self.sensor_mode == 'user':
                self.sensor_lengths.append(self.sensor_lengths_dict[user_id])
        
        if not self.final_user_ids:
            print("Warning: No users found in the intersection of all datasets. The dataset will be empty.")

    def __len__(self):
        return len(self.final_user_ids)

    def __getitem__(self, idx):
        user_id = self.final_user_ids[idx]
        
        # Start with the base survey data
        item = {
            'survey_features': self.survey_features_dict[user_id],
            'survey_labels': self.survey_labels_dict[user_id],
            'user_id': user_id
        }
        
        # Add diary data if requested
        if self.use_diary:
            item['diary_features'] = self.diary_features_dict[user_id]
            item['diary_length'] = self.diary_lengths_dict[user_id]
            
        # Add sensor data if requested
        if self.use_sensor:
            item['sensor_features'] = self.sensor_features_dict[user_id]
            if self.sensor_mode == 'user':
                item['sensor_length'] = self.sensor_lengths_dict[user_id]
        
        return item


def collate_fn_merged(batch):
    """
    A custom collate function for the MergedDataset.
    Takes a list of dictionary items and batches them.
    
    - Stacks static data (survey_features, survey_labels)
    - Pads sequence data (diary_features, sensor_features)
    """
    batch_out = {}
    
    # --- Stack Static Data ---
    # These are always present
    batch_out['survey_features'] = torch.stack([item['survey_features'] for item in batch])
    batch_out['survey_labels'] = torch.stack([item['survey_labels'] for item in batch])
    batch_out['user_ids'] = [item['user_id'] for item in batch]
    
    # --- Pad Diary Data (if present) ---
    if 'diary_features' in batch[0]:
        diary_features_list = [item['diary_features'] for item in batch]
        batch_out['diary_features'] = pad_sequence(diary_features_list, batch_first=True, padding_value=0.0)
        batch_out['diary_lengths'] = torch.tensor([item['diary_length'] for item in batch])

    # --- Pad Sensor Data (if present) ---
    if 'sensor_features' in batch[0]:
        sensor_features_list = [item['sensor_features'] for item in batch]
        batch_out['sensor_features'] = pad_sequence(sensor_features_list, batch_first=True, padding_value=0.0)
        
        # Lengths are only added in 'user' mode
        if 'sensor_length' in batch[0]:
            batch_out['sensor_lengths'] = torch.tensor([item['sensor_length'] for item in batch])
            
    return batch_out


# ---
# --- Wrapper Functions ---
# ---

def get_merged_survey_diary_loader(
    survey_csv, diary_csv,
    survey_args, diary_args,
    batch_size=2, shuffle=True
):
    """
    Returns a DataLoader for Survey + Sleep Diary data.
    """
    dataset = MergedDataset(
        survey_csv=survey_csv, diary_csv=diary_csv,
        survey_args=survey_args, diary_args=diary_args,
        use_diary=True, use_sensor=False
    )
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_merged
    )
    return loader

def get_merged_survey_sensor_loader(
    survey_csv, sensor_csv,
    survey_args, sensor_args,
    batch_size=2, shuffle=True
):
    """
    Returns a DataLoader for Survey + Sensor HRV data.
    
    Note: The `sensor_args` dictionary must specify the 'mode'
    (e.g., 'user' or 'window').
    - 'user' mode will use a BucketBatchSampler.
    - 'window' mode will use a standard DataLoader.
    """
    dataset = MergedDataset(
        survey_csv=survey_csv, sensor_csv=sensor_csv,
        survey_args=survey_args, sensor_args=sensor_args,
        use_diary=False, use_sensor=True
    )
    
    sensor_mode = sensor_args.get('mode', 'user')
    
    if sensor_mode == 'user':
        # --- User Mode: Use BucketBatchSampler on sensor lengths ---
        lengths = dataset.sensor_lengths
        bucket_sampler = BucketBatchSampler(lengths, batch_size, shuffle)
        
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=bucket_sampler, # Use batch_sampler
            collate_fn=collate_fn_merged
        )
    else: # 'window' mode
        # --- Window Mode: Standard DataLoader ---
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_merged
        )
        
    return loader

def get_merged_all_loader(
    survey_csv, diary_csv, sensor_csv,
    survey_args, diary_args, sensor_args,
    batch_size=2, shuffle=True
):
    """
    Returns a DataLoader for Survey + Sleep Diary + Sensor HRV data.
    
    Note: The `sensor_args` dictionary must specify the 'mode'
    (e.g., 'user' or 'window').
    - 'user' mode will use a BucketBatchSampler (sorted by *sensor* length).
    - 'window' mode will use a standard DataLoader.
    """
    dataset = MergedDataset(
        survey_csv=survey_csv, diary_csv=diary_csv, sensor_csv=sensor_csv,
        survey_args=survey_args, diary_args=diary_args, sensor_args=sensor_args,
        use_diary=True, use_sensor=True
    )
    
    sensor_mode = sensor_args.get('mode', 'user')
    
    if sensor_mode == 'user':
        # --- User Mode: Use BucketBatchSampler ---
        # We sort by sensor length, as it's typically the
        # most variable and longest sequence.
        lengths = dataset.sensor_lengths
        bucket_sampler = BucketBatchSampler(lengths, batch_size, shuffle)
        
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=bucket_sampler, # Use batch_sampler
            collate_fn=collate_fn_merged
        )
    else: # 'window' mode
        # --- Window Mode: Standard DataLoader ---
        # Note: Diary data will still be padded, but batches
        # are not bucketed by length.
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_merged
        )
        
    return loader