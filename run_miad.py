import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
import os
from tqdm import tqdm
from medpy import metric

from src.dataset import load_data, MedicalDataset, MedicalDatasetSeg
from src.utils import get_sam_model, get_prompt
from src.memory import PatchCore
from src.common import FaissNN, IdentitySampler, GreedyCoresetSampler, ApproximateGreedyCoresetSampler
from src.metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics


def main(args: argparse.Namespace) -> None:
    save = args.save
    _class_ = args.dataset
    set_imgs = 0
    img_to_plot = []
    predictor = get_sam_model("vit_b", args.device, args.med)
    train_dataloader, test_dataloader, use_mask = load_data(args.device, _class_, args.size)
    
    if not args.load:
        print('Extracting features from train dataset...')
        all_feat = []
        all_feat_multi = []#[[] for _ in range(12)]
        
        for img, _, _, _ in tqdm(train_dataloader):
            if set_imgs < 64:
                img_to_plot.append(img[0].cpu())
                set_imgs += 1
            if len(all_feat) > 10:
                break
            predictor.set_image(np.asarray(img[0].cpu()))
            features, all_features = predictor.get_image_embedding() #All_features = 12 blocks extracted
            all_feat.append(features.cpu())
            #all_features = [x.cpu() for x in all_features]
            #for layer in range(12):
            #    all_feat_multi[layer].append(all_features[layer])
            all_feat_multi.append(None)#all_features)

        #all_feat_multi = np.concatenate(all_feat_multi, axis=0)
        
        all_feat = np.concatenate(all_feat, axis=0)

        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        axes = axes.flatten()
        for i in range(len(img_to_plot)):
            axes[i].imshow(img_to_plot[i], cmap='gray')
            axes[i].axis('off')
        fig.savefig('results/train_out/train_set.png')

    layers_to_extract_from = None#[5,11]

    nn_method = FaissNN(True, 4)
    sampler = IdentitySampler()#ApproximateGreedyCoresetSampler(0.5, args.device)#If big dataset
    patchcore_instance = PatchCore(args.device)
    patchcore_instance.load(
        device=args.device,
        layers_to_extract_from=layers_to_extract_from,
        featuresampler=sampler,
        anomaly_scorer_num_nn=5,
        nn_method=nn_method,
        im_size=args.size
    )
    patchcore_save_path = os.path.join(
                        "models", _class_
                    )
    os.makedirs(patchcore_save_path, exist_ok=True)
    prepend = ''
    
    if args.load:
        print('Loading index....')
        patchcore_instance.load_from_path(patchcore_save_path, prepend)
    else:
        if layers_to_extract_from:
            patchcore_instance.fit(all_feat_multi)
        else:
            patchcore_instance.fit(torch.from_numpy(all_feat))
        
        patchcore_instance.save_to_path(patchcore_save_path, prepend)
    
        del all_feat, all_feat_multi
    torch.cuda.empty_cache()

    print('Extracting features from test dataset...')
    all_feat_test = []
    all_feat_multi_test = [[] for _ in range(12)]
    ite = 0
    labels_gt = []
    images_gt = []
    filenames = []
    masks_gt = []
    img_to_plot = []
    set_imgs = 0
    extract_idx = 2
    if use_mask:
        for img, gt, label, img_path in tqdm(test_dataloader):
            if set_imgs < 64 and label != 0:
                    img_to_plot.append(img[0].cpu())
                    set_imgs += 1
            #if len(labels_gt) >= 10:
            #    break
            labels_gt.append(label[0])
            masks_gt.append(gt[0])
            filenames.append(img_path[0])
            images_gt.append(np.asarray(img[0].cpu()))
            predictor.set_image(np.asarray(img[0].cpu()))
            features, all_features = predictor.get_image_embedding()
            if ite == extract_idx and save:
                plt.imsave('results/train_out/image.png', img[0].cpu().numpy())
                plt.imsave('results/train_out/gt.png', np.moveaxis(gt[0].cpu().numpy(), 0, -1)[:,:,0], cmap='gray')
                to_seg = np.asarray(img[0].cpu())
                save_feat = all_features#features.cpu()
            ite += 1
            #all_features = [x.cpu() for x in all_features]
            all_feat_test.append(features.cpu())
            #all_feat_multi_test.append(all_features)
            #for layer in range(12):
            #    all_feat_multi_test[layer].append(all_features[layer].cpu().numpy())
    #all_feat_multi_test = np.array(all_feat_multi_test)
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(len(img_to_plot)):
        axes[i].imshow(img_to_plot[i], cmap='gray')
        axes[i].axis('off')
    fig.savefig('results/train_out/test_set.png')
    all_feat_test = np.concatenate(all_feat_test, axis=0)
    print('Predicting...')
    
    if layers_to_extract_from:
        scores, segmentations = patchcore_instance.predict(torch.from_numpy(all_feat_multi_test))
    else:
        scores = [None] * len(all_feat_test)
        segmentations = [None] * len(all_feat_test)
        size_eval = 20
        for i in tqdm(range(0, len(all_feat_test), size_eval)):
            batch = all_feat_test[i:i + size_eval]
            sc, seg = patchcore_instance.predict(torch.from_numpy(batch))
            scores[i:i + size_eval] = sc
            segmentations[i:i + size_eval] = seg

    ''' Evaluation '''
    scores = np.array(scores)
    min_scores = scores.min(axis=-1)#.reshape(-1, 1)
    max_scores = scores.max(axis=-1)#.reshape(-1, 1)
    scores = (scores - min_scores) / (max_scores - min_scores)

    segmentations = np.array(segmentations)
    min_scores = np.min(segmentations)
    max_scores = np.max(segmentations)
    #print(max_scores.shape)
    segmentations = (segmentations - min_scores) / (max_scores - min_scores)


    anomaly_labels = [x != 0 for x in labels_gt]
    #print(anomaly_labels)
    
    auroc = compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
    )["auroc"]
    print('Auroc: {}'.format(auroc))
    
    if use_mask:
        # Compute PRO score & PW Auroc for all images
        pixel_scores = compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        full_pixel_auroc = pixel_scores["auroc"]
        print('Pixel Auroc: {}'.format(full_pixel_auroc))
        # Compute PRO score & PW Auroc only images with anomalies
        sel_idxs = []
        for i in range(len(masks_gt)):
            if anomaly_labels[i]:
                sel_idxs.append(i)
        pixel_scores = compute_pixelwise_retrieval_metrics(
            [segmentations[i] for i in sel_idxs],
            [masks_gt[i] for i in sel_idxs],
        )
        anomaly_pixel_auroc = pixel_scores["auroc"]
        print('Pro Auroc: {}'.format(anomaly_pixel_auroc))

    if save:
        for k in range(0,len(save_feat), 2):
            grid_size = int(np.sqrt(all_feat_test[0,0].shape[0]))
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            for i in range(grid_size):
                for j in range(grid_size):
                    axs[i, j].imshow(save_feat[k][0,i * grid_size + j].cpu(), cmap='gray')
                    axs[i, j].axis('off') 

            plt.subplots_adjust(wspace=0, hspace=0)
            fig.savefig('results/train_out/features_{}.png'.format(k+1))
        
        predictor.set_image(to_seg)
        bbs, point, mask_prompt, th = get_prompt(segmentations[extract_idx])
        plt.imsave('results/train_out/mask_prompt.png', segmentations[extract_idx], cmap='gray')
        total_mask0 = np.zeros_like(segmentations[0])
        total_mask1 = np.zeros_like(segmentations[0])
        total_mask2 = np.zeros_like(segmentations[0])
        for j, bb in enumerate(bbs):
            masks, confs, _ = predictor.predict(box=bb, mask_input=mask_prompt)
            total_mask0+=masks[0]
            total_mask1+=masks[1]
            total_mask2+=masks[2]
            #for i, (m, c) in enumerate(zip(masks, confs)):
            #    plt.imsave('out/segmentations_sam_{}_{}.png'.format(j,i), m, cmap='gray')
        plt.imsave('results/train_out/segmentation_prompt.png', th, cmap='gray')
        plt.imsave('results/train_out/segmentation0.png', total_mask0, cmap='gray')
        plt.imsave('results/train_out/segmentation1.png', total_mask1, cmap='gray')
        plt.imsave('results/train_out/segmentation2.png', total_mask2, cmap='gray')
        #plt.imsave('out/seg_diff.png', total_mask-segmentations[0], cmap='gray')
    if use_mask:
        sam_segmentations=[]
        sam_segmentations1 = []
        sam_segmentations2 = []
        prompts = []
        counter_normal = 0
        counter_abnormal = 0
        for im, la, gt, sg, sc, fn in zip(images_gt, anomaly_labels, masks_gt, segmentations, scores, filenames):
            predictor.set_image(im)
            bbs, point, mask_prompt, th = get_prompt(sg)
            prompts.append(th)
            total_mask0 = np.zeros_like(sg)
            total_mask1 = np.zeros_like(sg)
            total_mask2 = np.zeros_like(sg)
            if args.use_th :
                for j, bb in enumerate(bbs):
                    masks, confs, _ = predictor.predict(box=bb,mask_input=mask_prompt)
                    total_mask0+=masks[0]
                    total_mask1+=masks[1]
                    total_mask2+=masks[2]
            else:
                masks, confs, _ = predictor.predict(point_coords=np.asarray([point]), point_labels=np.asarray([1]))
                total_mask0+=masks[0]
                total_mask1+=masks[1]
                total_mask2+=masks[2]
            
            sam_segmentations.append(np.clip(total_mask0, 0, 1))
            sam_segmentations1.append(np.clip(total_mask1, 0, 1))
            sam_segmentations2.append(np.clip(total_mask2, 0, 1))
            if save and ((la and counter_abnormal < 100) or (not la and counter_normal < 20)):
                if not os.path.isdir('results/{}'.format(_class_)):
                    os.makedirs('results/{}'.format(_class_))
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                axes[0].imshow(im)
                axes[0].scatter([point[0]], [point[1]], color='red', s=10)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(np.moveaxis(gt.cpu().numpy(), 0, -1)[:,:,0], cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(sg, cmap='gray')
                axes[2].set_title("Anomaly Map")
                axes[2].axis("off")
                
                axes[3].imshow(total_mask0, cmap='gray')
                axes[3].set_title("Prediction 1")
                axes[3].axis("off")

                axes[4].imshow(total_mask2, cmap='gray')
                axes[4].set_title("Prediction 3")
                axes[4].axis("off")

                fig.suptitle('Score: {}'.format(sc), fontsize=16)

                plt.savefig('results/{}/out_{}'.format(_class_, os.path.basename(fn)), dpi=300, bbox_inches="tight")
                plt.close()

                if la:
                    counter_abnormal += 1
                else:
                    counter_normal += 1

        tot_dice = 0
        counter = 0
        print(masks_gt[0].shape)
        for i in range(len(masks_gt)):
            if anomaly_labels[i] != 0:
                dice = metric.binary.dc(prompts[i], masks_gt[i].cpu().detach().numpy())
                tot_dice += dice
                counter += 1
        tot_dice = tot_dice / counter
        print('Dice on th {} = {}'.format(counter,tot_dice))

        tot_dice = 0
        counter = 0
        for i in range(len(masks_gt)):
            if anomaly_labels[i] != 0:
                dice = metric.binary.dc(sam_segmentations[i], masks_gt[i].cpu().detach().numpy())
                tot_dice += dice
                counter += 1
        tot_dice = tot_dice / counter
        print('Dice SAM 0 = {}'.format(tot_dice))
        tot_dice = 0
        counter = 0
        for i in range(len(masks_gt)):
            if anomaly_labels[i] != 0:
                dice = metric.binary.dc(sam_segmentations1[i], masks_gt[i].cpu().detach().numpy())
                tot_dice += dice
                counter += 1
        tot_dice = tot_dice / counter
        print('Dice SAM 1 = {}'.format(tot_dice))
        
        tot_dice = 0
        counter = 0
        for i in range(len(masks_gt)):
            if anomaly_labels[i] != 0:
                dice = metric.binary.dc(sam_segmentations2[i], masks_gt[i].cpu().detach().numpy())
                tot_dice += dice
                counter += 1
        tot_dice = tot_dice / counter
        print('Dice SAM 2 = {}'.format(tot_dice))
    else:
        print('Impossible to compute pixelwise metrics for dataset \"{}\", missing GT masks'.format(_class_))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="The device to run on.")
    parser.add_argument("--dataset", type=str, default="RESC", choices=['RESC', 'BRAIN', 'LIVER'], help="Data type.")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--med", action="store_true")
    parser.add_argument("--use_th", action="store_true")
    
    args = parser.parse_args()

    if args.load == True:
        print('Evaluating {}'.format(args.dataset))
    else:
        print('Training {}'.format(args.dataset))

    main(args)