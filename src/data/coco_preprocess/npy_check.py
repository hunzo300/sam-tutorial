import os
import numpy as np


folder_path = "/mnt/sda/minkyukim/sam_dataset/coco_npy_train_dataset/gts"

npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

empty_files = []

for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)
    
    if np.all(data == 0):
        empty_files.append(npy_file)

if empty_files:
    print("0으로만 이루어진 npy 파일들:")
    for empty_file in empty_files:
        print(empty_file)
else:
    print("모든 npy 파일에 0이 아닌 값이 존재합니다.")


# 0으로만 이루어진 npy 파일들:
# 100226_3.npy
# 112614_3.npy
# 210804_1.npy
# 499198_12.npy
# 438629_5.npy
# 426400_6.npy
# 550395_8.npy
