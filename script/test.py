import autorootcwd
import torch
import torch.nn.functional as F
import numpy as np
import os
from script.train import NpyDataset, MedSAM, sam_model_registry, device, args, join
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.eval()

    iou_list = []
    dice_list = []

    for step, (image, gt, bboxes, names_temp) in enumerate(dataloader):
        image = image.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            
            pred_mask = medsam_model(image, bboxes.numpy())

       
        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized = F.interpolate(pred_mask.float(), size=(1024, 1024), mode='nearest').squeeze(1)

        iou, dice = calculate_metrics(pred_mask_resized, gt_resized)

        iou_list.append(iou)
        dice_list.append(dice)

        print(f"Image: {names_temp[0]}")
        print(f"IOU: {iou:.4f}, Dice: {dice:.4f}")

    avg_iou = np.mean(iou_list)
    avg_dice = np.mean(dice_list)

    print(f"Average IOU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")

if __name__ == "__main__":
    test()
