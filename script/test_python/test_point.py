import autorootcwd
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from script.train_ivdm_point import NpyDataset, MedSAM, sam_model_registry, device, args, join


def calculate_metrics(pred_mask, gt_mask):
    pred_mask = (pred_mask > 0.5).float()  # Thresholding to ensure binary mask
    gt_mask = (gt_mask > 0.5).float()  # Thresholding to ensure binary mask

    pred_flat = pred_mask.view(-1).float()
    gt_flat = gt_mask.view(-1).float()

    intersection = torch.sum(pred_flat * gt_flat)
    union = torch.sum(pred_flat) + torch.sum(gt_flat) - intersection

    epsilon = 1e-7  # To avoid division by zero

    iou = (intersection + epsilon) / (union + epsilon)
    dice = (2 * intersection + epsilon) / (torch.sum(pred_flat) + torch.sum(gt_flat) + epsilon)

    return iou.item(), dice.item()


def test():
    dataset = NpyDataset(args.tr_npy_path)
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Evaluation should be deterministic
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.eval()

    results = []

    for step, (image, gt, bboxes, point_coords, point_labels, names_temp) in enumerate(tqdm(test_dataloader)):
        image = image.to(device)
        gt = gt.to(device)
        bboxes = bboxes.cpu().numpy()  # Convert bounding boxes to numpy

        with torch.no_grad():
            pred_mask = medsam_model(image, bboxes)

        # Resize ground truth and predicted masks to the original image size (1024, 1024)
        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(1)

        # Calculate metrics for each image in the batch
        for idx in range(image.shape[0]):  # Iterate through batch
            iou, dice = calculate_metrics(pred_mask_resized[idx], gt_resized[idx])

            gt_file_name = os.path.basename(dataset.gt_path_files[step * args.batch_size + idx])
            image_name = '_'.join(gt_file_name.split('_')[:-1]) + '.npy'

            result = {
                "gt_file_name": gt_file_name,
                "image_name": image_name,
                "iou": iou,
                "dice": dice
            }
            results.append(result)

            print(f"Image: {image_name}")
            print(f"IOU: {iou:.4f}, Dice: {dice:.4f}")

    # Calculate average IOU and Dice
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])

    # Sort results by IOU and Dice to find the worst-performing images
    sorted_results = sorted(results, key=lambda x: (x['iou'], x['dice']))
    worst_two = sorted_results[:2]

    # Output data dictionary
    output_data = {
        "average_iou": avg_iou,
        "average_dice": avg_dice,
        "worst_images": worst_two,  # Worst performing images
        "dataset_path": args.tr_npy_path,
        "checkpoint_path": args.checkpoint
    }

    # Convert output data to DataFrame
    df = pd.DataFrame([output_data])
    csv_file_path = "/home/minkyukim/sam-tutorial/output_csv_point/test_results_point.csv"  # Change the output path if needed

    # Write results to CSV
    if not os.path.isfile(csv_file_path):
        df.to_csv(csv_file_path, index=False)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

    print("Results saved to", csv_file_path)


if __name__ == "__main__":
    test()
