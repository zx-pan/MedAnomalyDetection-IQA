import os
import csv
from sklearn.model_selection import KFold

# Define the directory paths
base_dir = '/afs/crc.nd.edu/user/z/zpan3/Datasets/Medical-Anomaly-Detection/t1ce/Brats21'
flair_dir = os.path.join(base_dir, 't1ce')

# Create a list of all files in the FLAIR directory
flair_files = [f for f in os.listdir(flair_dir) if f.endswith('_t1ce.nii.gz')]
flair_files.sort()  # Sort to ensure reproducibility

# Define the number of folds
num_folds = 5

# Initialize KFold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate over each fold
for fold, (train_val_idx, test_idx) in enumerate(kf.split(flair_files)):
    # Split into train+val and test sets
    train_val_files = [flair_files[i] for i in train_val_idx]
    test_files = [flair_files[i] for i in test_idx]

    # Further split train+val into train, val, and anomaly_val sets
    total_train_val = len(train_val_files)
    train_end_idx = int(total_train_val * 0.75)  # 60% of total files
    val_end_idx = int(total_train_val * 0.875)  # 10% of total files

    train_files = train_val_files[:train_end_idx]
    val_files = train_val_files[train_end_idx:val_end_idx]
    anomaly_val_files = train_val_files[val_end_idx:]

    # Write anomaly test files
    with open(os.path.join(base_dir, f'anomaly_test_fold_{fold}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_path', 'mask_path', 'seg_path'])
        for f in test_files:
            base_name = f.replace('_t1ce.nii.gz', '')
            img_path = os.path.join('t1ce', f)
            mask_path = os.path.join('mask', f'{base_name}_mask.nii.gz')
            seg_path = os.path.join('seg', f'{base_name}_seg.nii.gz')
            writer.writerow([img_path, mask_path, seg_path])

    # Write training files
    with open(os.path.join(base_dir, f'train_fold_{fold}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_path', 'mask_path', 'seg_path'])
        for f in train_files:
            base_name = f.replace('_t1ce.nii.gz', '')
            img_path = os.path.join('Healthy', f)
            mask_path = os.path.join('Healthy_mask', f'{base_name}_mask.nii.gz')
            seg_path = ''  # Set seg_path as null
            writer.writerow([img_path, mask_path, seg_path])

    # Write validation files
    with open(os.path.join(base_dir, f'val_fold_{fold}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_path', 'mask_path', 'seg_path'])
        for f in val_files:
            base_name = f.replace('_t1ce.nii.gz', '')
            img_path = os.path.join('Healthy', f)
            mask_path = os.path.join('Healthy_mask', f'{base_name}_mask.nii.gz')
            seg_path = ''  # Set seg_path as null
            writer.writerow([img_path, mask_path, seg_path])

    # Write anomaly validation files
    with open(os.path.join(base_dir, f'anomaly_val_fold_{fold}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['img_path', 'mask_path', 'seg_path'])
        for f in anomaly_val_files:
            base_name = f.replace('_t1ce.nii.gz', '')
            img_path = os.path.join('t1ce', f)
            mask_path = os.path.join('mask', f'{base_name}_mask.nii.gz')
            seg_path = os.path.join('seg', f'{base_name}_seg.nii.gz')
            writer.writerow([img_path, mask_path, seg_path])

print("5-fold cross-validation files have been created.")
