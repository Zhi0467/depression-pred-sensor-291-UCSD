"""
__version__: 0.0

This file provides three main "wrapper" functions to easily load and process
data from the survey, sleep diary, and sensor HRV CSV files:

1.  get_survey_data_loader(csv_file, ...):
    -   Returns: A standard DataLoader.
    -   Batches Yield: (features, labels, user_ids)
        -   features (Tensor): (batch_size, num_features)
        -   labels (Tensor): (batch_size, num_labels)
        -   user_ids (list[str]): [id1, id2, ...]

2.  get_sleep_diary_loader(csv_file, ...):
    -   Returns: A DataLoader using a custom collate function.
    -   Batches Yield: (features_padded, lengths, user_ids)
        -   features_padded (Tensor): (batch_size, max_seq_len, num_features)
        -   lengths (Tensor): (batch_size,) - Original sequence lengths.
        -   user_ids (list[str]): [id1, id2, ...]

3.  get_sensor_hrv_loader(csv_file, mode, ...):
    -   Mode 'user': Uses a BucketBatchSampler for efficient padding.
        -   Returns: A DataLoader using the bucket sampler and collate function.
        -   Batches Yield: (features_padded, lengths, user_ids)
            -   features_padded (Tensor): (batch_size, max_seq_len, num_features)
            -   lengths (Tensor): (batch_size,) - Original sequence lengths.
            -   user_ids (list[str]): [id1, id2, ...]
            
    -   Mode 'window': Uses a standard DataLoader for fixed-size windows.
        -   Returns: A standard DataLoader.
        -   Batches Yield: (features, user_ids)
            -   features (Tensor): (batch_size, window_size, num_features)
            -   user_ids (list[str]): [id1, id2, ...]

This file also contains the underlying Dataset classes (SurveyDataset,
SequenceDataset, SensorHRVDataset) and helper utilities (collate_fn_seq,
BucketBatchSampler) used by these wrappers.

Specifically, the SensorHRVDataset has two modes:
- user: 
    - all features of a given user form one tensor of shape (num_measured_intervals, num_features)
    - num_measured_intervals is different for each user
    - and in sensor_hrv.csv, they differ wildly
    - BucketBatchSampler is used to reduce excessive padding
    - users in one batch are padded to the same num_measured_intervals
    - batches differ in num_measured_intervals
- window:
    - the DataLoader returns (B, window_size, F)
    - user-agnostic
    - items in a batch and diffenrent batches have same window_size 

refer to:
- test.test_dataloaders.py to learn how to use the these classes

TODO: to implement an engineered version of any dataset, say diary data:
    - import collate_fn_seq in a new file: src/utils/engineered_dataloaders.py
    - implement desired feature columns, e.g. sleep duration variance
    - copy paste SequenceDataset, modify a few lines in init to add your features
    - copy paste get_sleep_diary_loader, change SequenceDataset to your class
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import random

# --- Base Datasets ---
class SurveyDataset(Dataset):
    """
    PyTorch Dataset for static, per-user data (e.g., survey.csv).
    Each row is treated as a separate sample.
    """
    def __init__(self, csv_file, id_col, feature_cols, label_cols=None, nan_fill_value=0.0, normalize=False):
        """
        Args:
            csv_file (str): Path to the csv file.
            id_col (str): Name of the column containing the user/device ID.
            feature_cols (list[str]): List of column names to be used as features.
            label_cols (list[str], optional): List of column names to be used as labels.
            nan_fill_value (float, optional): Value to fill NaNs with.
            normalize (bool, optional): Whether to apply StandardScaler to features.
        """
        if feature_cols is None:
            print("Warning: 'feature_cols' not provided to SurveyDataset. Using default list.")
            self.feature_cols = [
                'sex', 'age', 'marriage', 'smartwatch', 'regular', 
                'exercise', 'coffee', 'smoking', 'drinking', 'height', 'weight'
            ]
        else:
            self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.id_col = id_col

        df = pd.read_csv(csv_file)
        
        # Define all columns we need to process
        all_cols_to_process = self.feature_cols + (self.label_cols if self.label_cols else [])
        
        # Convert to numeric, coercing errors (e.g., strings) to NaN
        df[all_cols_to_process] = df[all_cols_to_process].apply(pd.to_numeric, errors='coerce')
        
        # Fill NaNs
        df = df.fillna(nan_fill_value)

        # Store IDs
        self.ids = df[self.id_col].values
        
        # Normalize features if requested
        if normalize and self.feature_cols:
            scaler = StandardScaler()
            df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])
            self.scaler = scaler # Save scaler if you need to inverse_transform later

        # Convert to tensors
        self.features = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
        
        if self.label_cols:
            self.labels = torch.tensor(df[self.label_cols].values, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        user_id = self.ids[idx]
        
        if self.labels is not None:
            label = self.labels[idx]
            return features, label, user_id
        else:
            return features, user_id

class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence data (e.g., sleep_diary.csv or sensor_hrv.csv).
    It groups all rows for a single user into one sample (a sequence).
    """
    def __init__(self, csv_file=None, dataframe=None, id_col=None, feature_cols=None, sort_by_col=None, nan_fill_value=0.0, normalize=False):
        """
        Args:
            csv_file (str, optional): Path to the csv file.
            dataframe (pd.DataFrame, optional): A pre-loaded DataFrame.
            id_col (str): Name of the column to group by (e.g., 'userId', 'deviceId').
            feature_cols (list[str]): List of column names to be used as features.
            sort_by_col (str, optional): Column to sort the sequence by (e.g., 'date', 'ts_start').
            nan_fill_value (float, optional): Value to fill NaNs with.
            normalize (bool, optional): Whether to apply StandardScaler to features.
        """
        if feature_cols is None:
            print("Warning: 'feature_cols' not provided to SequenceDataset. Using default list.")
            self.feature_cols = [
                'sleep_duration', 'in_bed_duration', 'sleep_latency', 
                'sleep_efficiency', 'waso', 'wakeup@night'
            ]
        else:
            self.feature_cols = feature_cols
        self.id_col = id_col

        if dataframe is not None:
            df = dataframe.copy() # Use copy to avoid side effects
        elif csv_file is not None:
            df = pd.read_csv(csv_file)
        else:
            raise ValueError("Either 'csv_file' or 'dataframe' must be provided.")

        # Define all columns we need to process
        all_cols_to_process = self.feature_cols
        
        # Ensure sort_by_col is handled, even if not a feature/label
        if sort_by_col and sort_by_col not in df.columns:
            print(f"Warning: sort_by_col '{sort_by_col}' not in DataFrame columns.")
            sort_by_col = None

        # Convert data columns to numeric, coercing errors to NaN
        df[all_cols_to_process] = df[all_cols_to_process].apply(pd.to_numeric, errors='coerce')
        
        # Fill NaNs
        df = df.fillna(nan_fill_value)

        # Sort sequences if requested
        if sort_by_col:
            if sort_by_col in df.columns:
                 # Try to convert sort column to numeric (like timestamps)
                try:
                    df[sort_by_col] = pd.to_numeric(df[sort_by_col])
                except ValueError:
                    # Handle the non-numeric values here. 
                    # Example: convert non-numeric to NaN
                    df[sort_by_col] = pd.to_numeric(df[sort_by_col], errors='coerce')
                df = df.sort_values(by=[self.id_col, sort_by_col])
            else:
                print(f"Warning: sort_by_col '{sort_by_col}' not found. Data will not be sorted.")

        # Normalize features if requested
        if normalize and self.feature_cols:
            scaler = StandardScaler()
            df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])
            self.scaler = scaler

        # Group by user and store sequences
        self.user_ids = []
        self.data_groups = []
        
        grouped = df.groupby(self.id_col)
        
        for user_id, group in grouped:
            self.user_ids.append(user_id)
            
            features = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
            
            self.data_groups.append(features)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        features = self.data_groups[idx]
        user_id = self.user_ids[idx]
        
        return features, user_id

def collate_fn_seq(batch):
    """
    Custom collate function for the SequenceDataset.
    It pads sequences to the same length within a batch.
    
    Args:
        batch (list): A list of tuples from SequenceDataset.__getitem__.
                      e.g., [(features1, labels1, id1), (features2, labels2, id2), ...]
                      or [(features1, id1), (features2, id2), ...]
    
    Returns:
        A tuple containing:
        - features_padded (Tensor): Padded features (batch_size, max_seq_len, num_features)
        - labels_padded (Tensor or None): Padded labels (batch_size, max_seq_len, num_labels)
        - lengths (Tensor): Original sequence lengths (batch_size,)
        - ids (list): List of user IDs
    """
    # Check if labels are present
    # __getitem__ now returns (features, id), so len is 2.
    # This logic correctly handles the no-label case.
    has_labels = (len(batch[0]) == 3)
    
    # Unzip the batch
    if has_labels:
        features_list, labels_list, ids_list = zip(*batch)
    else:
        features_list, ids_list = zip(*batch)
        labels_list = None

    # Store original lengths
    lengths = torch.tensor([len(f) for f in features_list])

    # Pad features
    # batch_first=True makes the output (batch_size, max_seq_len, num_features)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    # Pad labels if they exist
    labels_padded = None
    if labels_list:
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=0.0)
        return features_padded, labels_padded, lengths, list(ids_list)
    else:
        return features_padded, lengths, list(ids_list)

class BucketBatchSampler(Sampler):
    """
    A custom Sampler that groups sequences of similar lengths into batches.
    
    This is used to minimize padding in variable-length sequence models.
    """
    def __init__(self, lengths, batch_size, shuffle=True):
        """
        Args:
            lengths (list[int]): A list of lengths for each sample in the dataset.
            batch_size (int): The number of samples to include in each batch.
            shuffle (bool): Whether to shuffle the batches.
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create a list of (index, length) tuples
        indices = [(i, self.lengths[i]) for i in range(len(self.lengths))]
        
        # Sort by length
        indices.sort(key=lambda x: x[1])
        
        # Group sorted indices into batches
        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            self.batches.append([idx for (idx, length) in indices[i : i + self.batch_size]])
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

class SensorHRVDataset(Dataset):
    """
    A specialized Dataset for the Sensor HRV data.
    
    Operates in two modes:
    - 'user': (Default) One sample = one user. Returns a variable-length sequence.
              Requires `collate_fn_seq` for padding.
    - 'window': One sample = one fixed-size window (e.g., 300 rows).
                Does NOT require padding.
    """
    def __init__(self, csv_file, mode='user', window_size=300, step_size=300,
                 feature_cols=None, normalize=False, nan_fill_value=0.0):
        
        if csv_file is None:
            raise ValueError("'csv_file' must be provided.")
        
        self.mode = mode
        self.window_size = window_size
        self.step_size = step_size
            
        df = pd.read_csv(csv_file)
        
        id_col = 'deviceId'
        sort_by_col = 'ts_start'
        
        # Use the new default features list
        default_features = ['missingness_score', 'HR', 'ibi', 'steps', 'calories', 
            'light_avg', 'sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'lf', 'hf', 'lf/hf']

        # Allow user to override the default engineered features/labels
        final_features = feature_cols if feature_cols is not None else default_features
        
        self.feature_cols = final_features

        # --- Data Preparation ---
        
        # Convert all relevant columns to numeric and fill NaNs
        all_cols_to_process = self.feature_cols
        df[all_cols_to_process] = df[all_cols_to_process].apply(pd.to_numeric, errors='coerce')
        df = df.fillna(nan_fill_value)
        
        # Sort by user and timestamp
        try:
            # Try converting to numeric
            df[sort_by_col] = pd.to_numeric(df[sort_by_col])
        except ValueError:
            # Handle non-numeric values (e.g., if they are already datetimes or strings)
            # Coerce errors to NaN, which will be handled by sorting
            df[sort_by_col] = pd.to_numeric(df[sort_by_col], errors='coerce')
            
        df = df.sort_values(by=[id_col, sort_by_col])

        # Normalize features if requested
        if normalize and self.feature_cols:
            scaler = StandardScaler()
            df[self.feature_cols] = scaler.fit_transform(df[self.feature_cols])
            self.scaler = scaler

        # Group by user and create sequences based on mode
        self.data_groups = []
        self.lengths = [] # Store lengths for the BucketBatchSampler
        
        grouped = df.groupby(id_col)
        
        for user_id, group in grouped:
            
            if self.mode == 'user':
                # --- Mode 'user': One sequence per user ---
                features = torch.tensor(group[self.feature_cols].values, dtype=torch.float32)
                
                self.data_groups.append((features, user_id))
                self.lengths.append(len(features)) # Store the length
            
            elif self.mode == 'window':
                # --- Mode 'window': Fixed-size windows per user ---
                # Iterate over the user's group data in windows
                for i in range(0, len(group) - self.window_size + 1, self.step_size):
                    
                    feature_window = group[self.feature_cols].iloc[i : i + self.window_size].values
                    features = torch.tensor(feature_window, dtype=torch.float32)
                    
                    self.data_groups.append((features, user_id))
                    # No need to store length for window, as it's constant
                    # But we add a placeholder for consistency if needed, though sampler isn't used.
                    self.lengths.append(self.window_size) 

            else:
                raise ValueError(f"Unknown mode '{self.mode}'. Must be 'user' or 'window'.")
    
    def __len__(self):
        return len(self.data_groups)

    def __getitem__(self, idx):
        features, user_id = self.data_groups[idx]
        
        return features, user_id

# --- DataLoader Wrappers ---

def get_survey_data_loader(csv_file, id_col, feature_cols, label_cols=None, normalize=False, nan_fill_value=0.0, batch_size=2, shuffle=True):
    """
    Wrapper function to get a DataLoader for SurveyDataset.
    """
    survey_dataset = SurveyDataset(
        csv_file=csv_file,
        id_col=id_col,
        feature_cols=feature_cols,
        label_cols=label_cols,
        nan_fill_value=nan_fill_value,
        normalize=normalize
    )
    survey_loader = DataLoader(
        dataset=survey_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return survey_loader

def get_sleep_diary_loader(csv_file, id_col='userId', feature_cols=None, sort_by_col='date', normalize=False, nan_fill_value=0.0, batch_size=2, shuffle=True):
    """
    Wrapper function to get a DataLoader for the Sleep Diary SequenceDataset.
    Uses the base SequenceDataset.
    """
    
    # Define default features if none provided
    if feature_cols is None:
        feature_cols = ['sleep_duration', 'in_bed_duration', 'sleep_latency', 
                        'sleep_efficiency', 'waso', 'wakeup@night']
    
    sleep_dataset = SequenceDataset(
        csv_file=csv_file,
        id_col=id_col,
        feature_cols=feature_cols,
        sort_by_col=sort_by_col,
        nan_fill_value=nan_fill_value,
        normalize=normalize
    )
    sleep_loader = DataLoader(
        dataset=sleep_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_seq # Important for sequence padding
    )
    return sleep_loader

def get_sensor_hrv_loader(csv_file, mode='user', window_size=100, step_size=50, 
                          feature_cols=None, 
                          normalize=False, nan_fill_value=0.0, 
                          batch_size=2, shuffle=True):
    """
    Wrapper function to get a DataLoader for the Sensor HRV SequenceDataset.
    Uses the specialized SensorHRVDataset.
    
    Args:
        mode (str): 'user' (variable-length seqs) or 'window' (fixed-length seqs).
        window_size (int): Number of rows per window (if mode='window').
        step_size (int): Step between windows (if mode='window').
    """
    hrv_dataset = SensorHRVDataset(
        csv_file=csv_file,
        mode=mode,
        window_size=window_size,
        step_size=step_size,
        feature_cols=feature_cols, # User can override defaults
        nan_fill_value=nan_fill_value,
        normalize=normalize
    )
    
    # Conditionally set collate_fn:
    # - 'user' mode needs padding
    # - 'window' mode does not
    if mode == 'user':
        # --- User Mode: Use BucketBatchSampler for efficient padding ---
        
        # Get the lengths of all sequences
        lengths = hrv_dataset.lengths
        
        # Create the sampler
        bucket_sampler = BucketBatchSampler(lengths, batch_size, shuffle)
        
        hrv_loader = DataLoader(
            dataset=hrv_dataset,
            batch_sampler=bucket_sampler, # Use batch_sampler
            collate_fn=collate_fn_seq     # Still need collate_fn to pad the batches
            # When using batch_sampler, 'batch_size', 'shuffle', and 'sampler'
            # must be left as default (or None/False).
        )
    
    else: # mode == 'window'
        # --- Window Mode: Standard DataLoader ---
        # No collate_fn needed as all samples are the same size
        hrv_loader = DataLoader(
            dataset=hrv_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=None # All windows are the same size
        )
        
    return hrv_loader