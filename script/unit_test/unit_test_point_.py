import autorootcwd
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torch.nn.functional as F

from script.train_ivdm_point import NpyDataset, show_mask, show_box, MedSAM, sam_model_registry, device, args, join

def unit_test():

    dataset = NpyDataset("/mnt/sda/minkyukim/sam_dataset/coco_npy_test_dataset_1024image")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    sam_model_1 = sam_model_registry[args.model_type](checkpoint="/home/minkyukim/sam-tutorial/work_dir/SAM/sam_vit_b_01ec64.pth")
    medsam_model_1 = MedSAM(
        image_encoder=sam_model_1.image_encoder,
        mask_decoder=sam_model_1.mask_decoder,
        prompt_encoder=sam_model_1.prompt_encoder,
    ).to(device)
    medsam_model_1.eval()


    sam_model_2 = sam_model_registry[args.model_type](checkpoint="/mnt/sda/minkyukim/pth/sam-tutorial_coco/medsam_model_best_point.pth")
    medsam_model_2 = MedSAM(
        image_encoder=sam_model_2.image_encoder,
        mask_decoder=sam_model_2.mask_decoder,
        prompt_encoder=sam_model_2.prompt_encoder,
    ).to(device)
    medsam_model_2.eval()


    for step, (image, gt, bboxes, point_coords, point_labels, names_temp) in enumerate(dataloader):

        image = image.to(device)
        gt = gt.to(device)
        point_coords = point_coords.to(device)
        point_labels = point_labels.to(device)

        with torch.no_grad():
            bboxes_np = bboxes.detach().cpu().numpy()
            points_np = point_coords.detach().cpu().numpy()
            point_labels_np = point_labels.detach().cpu().numpy()
            pred_mask_1 = medsam_model_1(image, bboxes_np, (points_np, point_labels_np))
            pred_mask_2 = medsam_model_2(image, bboxes_np, (points_np, point_labels_np))

        os.makedirs("gt_images", exist_ok=True)

        gt_resized = F.interpolate(gt.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        gt_resized_np = gt_resized[0].cpu().numpy()
        gt_resized_np = np.clip(gt_resized_np, 0, 1)  

        plt.figure(figsize=(20, 5)) 


        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1) 
        plt.subplot(1, 4, 1)
        plt.imshow(image_np)
        show_box(bboxes[0].cpu().numpy(), plt.gca())
        fg_points = point_coords[point_labels == 1].cpu().numpy()  # Foreground points
        bg_points = point_coords[point_labels == 0].cpu().numpy()  # Background points
        plt.scatter(fg_points[:, 0], fg_points[:, 1], color='green', label='Foreground Points', marker='o', s=100)
        plt.scatter(bg_points[:, 0], bg_points[:, 1], color='red', label='Background Points', marker='x', s=100)
        plt.legend()
        plt.title("Original Image with BBox and Points")

        # GT Mask
        plt.subplot(1, 4, 2)
        plt.imshow(image_np)
        show_mask(gt_resized_np, plt.gca())
        plt.title("GT Mask")

        # Predicted Mask (Model 1)
        pred_mask_resized_1 = F.interpolate(pred_mask_1.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np_1 = pred_mask_resized_1[0].cpu().numpy()
        pred_mask_resized_np_1 = np.clip(pred_mask_resized_np_1, 0, 1)
        plt.subplot(1, 4, 3)
        plt.imshow(image_np)
        show_mask(pred_mask_resized_np_1, plt.gca())
        plt.title("Predicted Mask (pre-trained)")

        # Predicted Mask (Model 2)
        pred_mask_resized_2 = F.interpolate(pred_mask_2.float(), size=(1024, 1024), mode='nearest').squeeze(1)
        pred_mask_resized_np_2 = pred_mask_resized_2[0].cpu().numpy()
        pred_mask_resized_np_2 = np.clip(pred_mask_resized_np_2, 0, 1)
        plt.subplot(1, 4, 4)
        plt.imshow(image_np)
        show_mask(pred_mask_resized_np_2, plt.gca())
        plt.title("Predicted Mask (finetuned)")

        plt.savefig(f"output_images/coco_point_{names_temp[0]}_comparison.png", bbox_inches="tight", dpi=300)
        plt.close()

        break

if __name__ == "__main__":
    unit_test()
