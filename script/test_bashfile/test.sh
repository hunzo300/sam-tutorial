#!/bin/bash


CHECKPOINTS=("/home/minkyukim/sam-tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done
CHECKPOINTS=("/mnt/sda/minkyukim/pth/sam-tutorial_coco/medsam_model_best_new.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done
CHECKPOINTS=("/home/minkyukim/sam-tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done

CHECKPOINTS=("/mnt/sda/minkyukim/pth/sam-tutorial_brats/medsam_model_best_new.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/brats_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done

CHECKPOINTS=("/home/minkyukim/sam-tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/ivdm_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done

CHECKPOINTS=("/mnt/sda/minkyukim/pth/sam-tutorial_ivdm/medsam_model_best_new.pth")
DATASETS=("/mnt/sda/minkyukim/sam_dataset/ivdm_npy_test_dataset_1024image")

# Output file to store the results
OUTPUT_FILE="/home/minkyukim/sam-tutorial/output_csv/test_results.txt"

# Loop through each combination of checkpoint and dataset
for checkpoint in "${CHECKPOINTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "Testing with checkpoint: $checkpoint and dataset: $dataset"
        python /home/minkyukim/sam-tutorial/script/test_.py --tr_npy_path $dataset -checkpoint $checkpoint --device "cuda:3">> $OUTPUT_FILE
        echo "-----------------------------------" >> $OUTPUT_FILE
    done
done