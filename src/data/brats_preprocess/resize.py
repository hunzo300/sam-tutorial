import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_npy_files(input_dir):
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for npy_file in tqdm(npy_files, desc=f"Processing {input_dir}"):
        file_path = os.path.join(input_dir, npy_file)

        # Load the numpy array
        img = np.load(file_path)

        # Check if the image is already in the desired shape
        if img.shape != (1024, 1024, 3):
            # Resize the image to (1024, 1024, 3)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))  # Convert to PIL Image for resizing
            img_pil_resized = img_pil.resize((1024, 1024))
            img = np.array(img_pil_resized, dtype=np.float32) / 255.0  # Convert back to numpy array and normalize to [0, 1]

        # Normalize the image to [0, 1] if it's not already
        if np.max(img) > 1.0 or np.min(img) < 0.0:
            img = img / np.max(img)

        # Save the processed numpy array back to the same file
        np.save(file_path, img)

# Directories for train and test datasets
train_imgs_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_train_dataset/imgs'
test_imgs_dir = '/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset/imgs'

# Process the train dataset
process_npy_files(train_imgs_dir)

# Process the test dataset
process_npy_files(test_imgs_dir)

print("Processing complete.")
