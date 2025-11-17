import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src', 'utils')
sys.path.insert(0, project_root)
try:
    from src.utils.dataloaders import get_survey_data_loader, get_sensor_hrv_loader, get_sleep_diary_loader
except ImportError:
    print(f"Error: Could not import data_loaders from {src_path}")
    print("Please ensure 'src/utils/data_loaders.py' exists and this script is run from the project root.")
    sys.exit(1)

survey_csv_path = "data/survey.csv"
sleep_csv_path = "data/sleep_diary.csv"
hrv_csv_path = "data/sensor_hrv.csv"

print("--- 1. Loading Survey Dataset ---")
# Example: Predict final PHQ9 score from demographic and behavior data
survey_features = ['age', 'sex', 'marriage', 'occupation', 'smartwatch', 
                   'regular', 'exercise', 'coffee', 'smoking', 'drinking', 
                   'height', 'weight']
survey_labels = ['PHQ9_F', 'GAD7_F', 'ISI_F']

# Use the wrapper function
survey_loader = get_survey_data_loader(
    csv_file=survey_csv_path,
    id_col='deviceId',
    feature_cols=survey_features,
    label_cols=survey_labels,
    normalize=True,
    batch_size=2,
    shuffle=True
)

# Get one batch
features, labels, user_ids = next(iter(survey_loader))
print(f"Batch of survey data:")
print(f"  User IDs: {user_ids}")
print(f"  Features shape: {features.shape}")
print(f"  Labels shape: {labels.shape} (batch_size, num_labels)\n")


print("--- 2. Loading Sleep Diary Sequence Dataset ---")
# Using the wrapper, which uses the base SequenceDataset
sleep_loader = get_sleep_diary_loader(
    csv_file=sleep_csv_path,
    id_col='userId',
    feature_cols=None, # Use defaults
    normalize=False,
    batch_size=4,
    shuffle=True
)

# Get one batch
# Note: no 'labels_padded' since we set label_cols=None
features_padded, lengths, user_ids = next(iter(sleep_loader))
print(f"Batch of sleep diary data:")
print(f"  User IDs: {user_ids}")
print(f"  Padded Features shape: {features_padded.shape} (batch_size, max_seq_len, num_features)")
print(f"  Original sequence lengths: {lengths}\n")


print("--- 3. Loading Sensor HRV Sequence Dataset (Mode: 'user') ---")
# Use the wrapper with mode='user'
# This will use the new SensorHRVDataset, load engineered features,
# and apply padding.
hrv_loader_user = get_sensor_hrv_loader(
    csv_file=hrv_csv_path,
    mode='user',
    feature_cols=None, # Use engineered defaults
    normalize=True,
    batch_size=4,
    shuffle=True
)

# Get one batch
features_padded, lengths, user_ids = next(iter(hrv_loader_user))
print(f"Batch of sensor HRV data (mode='user'):")
print(f"  User IDs: {user_ids}")
print(f"  Padded Features shape: {features_padded.shape} (e.g., B, T, F)")
print(f"  Original feature sequence lengths for each user: {lengths}\n")


print("--- 4. Loading Sensor HRV Sequence Dataset (Mode: 'window') ---")
# Use the wrapper with mode='window'
# This creates fixed-size, non-padded sequences.
hrv_loader_window = get_sensor_hrv_loader(
    csv_file=hrv_csv_path,
    mode='window',
    window_size=100, 
    step_size=50,  
    feature_cols=None, # Use engineered defaults
    normalize=True,
    batch_size=4, # Only 1 sample will be generated
    shuffle=True
)

# Get one batch
# Note: No 'lengths' tensor returned, as padding is not used.
features_window, user_ids = next(iter(hrv_loader_window))
print(f"Batch of sensor HRV data (mode='window'):")
print(f"  User IDs: {user_ids}")
print(f"  Windowed Features shape: {features_window.shape} (B, window_size, F)")