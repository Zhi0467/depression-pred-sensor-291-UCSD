import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src', 'utils')
sys.path.insert(0, project_root)
try:
    from src.utils.merged_dataloaders import get_merged_all_loader, get_merged_survey_diary_loader, get_merged_survey_sensor_loader
except ImportError:
    print(f"Error: Could not import data_loaders from {src_path}")
    print("Please ensure 'src/utils/data_loaders.py' exists and this script is run from the project root.")
    sys.exit(1)

survey_csv_path = "data/survey.csv"
sleep_csv_path = "data/sleep_diary.csv"
hrv_csv_path = "data/sensor_hrv.csv"


# --- Define Argument Dictionaries ---

survey_args = {
    'id_col': 'deviceId',
    'feature_cols': None,
    'label_cols': ['PHQ9_F', 'GAD7_F', 'ISI_F'],
    'normalize': True
}

diary_args = {
    'id_col': 'userId',
    'feature_cols': None,
    'sort_by_col': 'date',
    'normalize': True
}

sensor_args_user_mode = {
    'mode': 'user',
    'feature_cols': None, # Use defaults
    'normalize': True
}

sensor_args_window_mode = {
    'mode': 'window',
    'window_size': 100, # Small window for dummy data
    'step_size': 50,
    'feature_cols': None,
    'normalize': True
}

# --- 1. Test Survey + Diary ---
print("\n--- 1. Testing Survey + Diary Loader ---")
loader_sd = get_merged_survey_diary_loader(
    survey_csv=survey_csv_path, diary_csv=sleep_csv_path,
    survey_args=survey_args, diary_args=diary_args,
    batch_size=4, shuffle=True
)

try:
    batch_sd = next(iter(loader_sd))
    print(f"Batch keys: {batch_sd.keys()}")
    print(f"User IDs: {batch_sd['user_ids']}")
    print(f"Survey Features shape: {batch_sd['survey_features'].shape}")
    print(f"Survey Labels shape: {batch_sd['survey_labels'].shape}")
    print(f"Diary Features shape: {batch_sd['diary_features'].shape}")
    print(f"Diary Lengths: {batch_sd['diary_lengths']}")
except StopIteration:
    print("Loader is empty. This might be correct if no users overlap.")
except Exception as e:
    print(f"Error iterating loader: {e}")


# --- 2. Test Survey + Sensor (User Mode) ---
print("\n--- 2. Testing Survey + Sensor Loader (User Mode) ---")
loader_ss_user = get_merged_survey_sensor_loader(
    survey_csv=survey_csv_path, sensor_csv=hrv_csv_path,
    survey_args=survey_args, sensor_args=sensor_args_user_mode,
    batch_size=4, shuffle=True
)

try:
    batch_ss_user = next(iter(loader_ss_user))
    print(f"Batch keys: {batch_ss_user.keys()}")
    print(f"User IDs: {batch_ss_user['user_ids']}")
    print(f"Survey Features shape: {batch_ss_user['survey_features'].shape}")
    print(f"Sensor Features shape: {batch_ss_user['sensor_features'].shape}")
    print(f"Sensor Lengths: {batch_ss_user['sensor_lengths']}")
except StopIteration:
    print("Loader is empty. This might be correct if no users overlap.")
except Exception as e:
    print(f"Error iterating loader: {e}")


# --- 3. Test Survey + Diary + Sensor (User Mode) ---
print("\n--- 3. Testing Survey + Diary + Sensor Loader (User Mode) ---")
loader_all = get_merged_all_loader(
    survey_csv=survey_csv_path, diary_csv=sleep_csv_path, sensor_csv=hrv_csv_path,
    survey_args=survey_args, diary_args=diary_args, sensor_args=sensor_args_user_mode,
    batch_size=8, shuffle=True
)

try:
    batch_all = next(iter(loader_all))
    print(f"Batch keys: {batch_all.keys()}")
    print(f"User IDs: {batch_all['user_ids']}")
    print(f"Survey Features shape: {batch_all['survey_features'].shape}")
    print(f"Diary Features shape: {batch_all['diary_features'].shape}")
    print(f"Diary Lengths: {batch_all['diary_lengths']}")
    print(f"Sensor Features shape: {batch_all['sensor_features'].shape}")
    print(f"Sensor Lengths: {batch_all['sensor_lengths']}")
except StopIteration:
    print("Loader is empty. This might be correct if no users overlap.")
except Exception as e:
    print(f"Error iterating loader: {e}")

# --- 4. Test Survey + Sensor (Window Mode) ---
print("\n--- 4. Testing Survey + Sensor Loader (Window Mode) ---")
loader_ss_window = get_merged_survey_sensor_loader(
    survey_csv=survey_csv_path, sensor_csv=hrv_csv_path,
    survey_args=survey_args, sensor_args=sensor_args_window_mode,
    batch_size=4, shuffle=False
)

try:
    batch_ss_window = next(iter(loader_ss_window))
    print(f"Batch keys: {batch_ss_window.keys()}")
    print(f"User IDs: {batch_ss_window['user_ids']}")
    print(f"Survey Features shape: {batch_ss_window['survey_features'].shape}")
    print(f"Sensor Features shape: {batch_ss_window['sensor_features'].shape}")
    print(f"Sensor shape should be (B, window_size, F)")
    # Note: No 'sensor_lengths' key in window mode
    print(f"Has 'sensor_lengths' key: {'sensor_lengths' in batch_ss_window}")
except StopIteration:
    print("Loader is empty. This is expected if no users meet window criteria.")
except Exception as e:
    print(f"Error iterating loader: {e}")