import os
import numpy as np
import nibabel as nib

# Define the directory paths
base_dir = '/afs/crc.nd.edu/user/z/zpan3/Datasets/Medical-Anomaly-Detection/T1CE/Brats21'
flair_dir = os.path.join(base_dir, 't1ce')
seg_dir = os.path.join(base_dir, 'seg')
mask_dir = os.path.join(base_dir, 'mask')
healthy_dir = os.path.join(base_dir, 'Healthy')
healthy_mask_dir = os.path.join(base_dir, 'Healthy_mask')

# Create the Healthy directory if it doesn't exist
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(healthy_mask_dir, exist_ok=True)

# List all files in the FLAIR directory
flair_files = [f for f in os.listdir(flair_dir) if f.endswith('.nii.gz')]

# Process each FLAIR file
for flair_file in flair_files:
    print(f"Processing {flair_file}...")

    # Derive the corresponding segmentation file name
    base_name = flair_file.replace('_t1ce.nii.gz', '')
    seg_file = f"{base_name}_seg.nii.gz"
    mask_file = f"{base_name}_mask.nii.gz"

    flair_path = os.path.join(flair_dir, flair_file)
    seg_path = os.path.join(seg_dir, seg_file)
    mask_path = os.path.join(mask_dir, mask_file)

    # Check if the corresponding segmentation file exists
    if not os.path.exists(seg_path):
        print(f"Segmentation file for {flair_file} not found, skipping.")
        continue

    # Load the FLAIR image and the corresponding segmentation file
    flair_img = nib.load(flair_path)
    seg_img = nib.load(seg_path)
    mask_img = nib.load(mask_path)

    flair_data = flair_img.get_fdata()
    seg_data = seg_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # Identify the healthy slices (slices with no tumor in the segmentation)
    healthy_slices = []
    healthy_mask_slices = []
    for i in range(flair_data.shape[2]):
        flair_slice = flair_data[:, :, i]
        seg_slice = seg_data[:, :, i]
        mask_slice = mask_data[:, :, i]

        if np.all(seg_slice == 0):  # Check for healthy or empty slices
            healthy_slices.append(flair_slice)
            healthy_mask_slices.append(mask_slice)

    # Stack the healthy slices along the third dimension
    if healthy_slices:
        print(f"Found {len(healthy_slices)} healthy slices in {flair_file}")

        healthy_data = np.stack(healthy_slices, axis=2)
        healthy_mask_data = np.stack(healthy_mask_slices, axis=2)

        # Create a new Nifti1Image for the healthy slices
        healthy_img = nib.Nifti1Image(healthy_data, flair_img.affine, flair_img.header)
        healthy_mask_img = nib.Nifti1Image(healthy_mask_data, mask_img.affine, mask_img.header)

        # Save the healthy image to the Healthy directory
        healthy_path = os.path.join(healthy_dir, flair_file)
        healthy_mask_path = os.path.join(healthy_mask_dir, mask_file)
        nib.save(healthy_img, healthy_path)
        nib.save(healthy_mask_img, healthy_mask_path)
    else:
        print(f"No healthy slices found in {flair_file}")

print("Processing complete.")
