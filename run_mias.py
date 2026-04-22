import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm

from src.common import FaissNN, IdentitySampler
from src.dataset import load_data
from src.memory import PatchCore
from src.metrics import (
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.utils import get_prompt, get_sam_model


PREVIEW_GRID = 64               # 8x8 preview grids
PREDICT_BATCH = 20
SAMPLE_IDX = 2                  # test-set index used for feature/prompt visualizations
NUM_SAM_MASKS = 3               # SAM returns 3 masks per prompt
MAX_SAVED_NORMAL = 20
MAX_SAVED_ABNORMAL = 100
TRAIN_OUT_DIR = 'results/train_out'


def save_image_grid(images, path, grid_side=8):
    fig, axes = plt.subplots(grid_side, grid_side, figsize=(10, 10))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    fig.savefig(path)
    plt.close(fig)


def extract_train_features(predictor, dataloader, preview_path):
    print('Extracting features from train dataset...')
    features, preview = [], []
    for img, _, _, _ in tqdm(dataloader):
        if len(preview) < PREVIEW_GRID:
            preview.append(img[0].cpu())
        predictor.set_image(np.asarray(img[0].cpu()))
        feat, _ = predictor.get_image_embedding()
        features.append(feat.cpu())
    save_image_grid(preview, preview_path)
    return np.concatenate(features, axis=0)


def extract_test_features(predictor, dataloader, preview_path, save):
    print('Extracting features from test dataset...')
    features, labels, masks, images, filenames, preview = [], [], [], [], [], []
    sample = {'to_seg': None, 'features': None}
    for idx, (img, gt, label, img_path) in enumerate(tqdm(dataloader)):
        if len(preview) < PREVIEW_GRID and label != 0:
            preview.append(img[0].cpu())
        labels.append(label[0])
        masks.append(gt[0])
        filenames.append(img_path[0])
        images.append(np.asarray(img[0].cpu()))
        predictor.set_image(np.asarray(img[0].cpu()))
        feat, all_feats = predictor.get_image_embedding()
        if idx == SAMPLE_IDX and save:
            plt.imsave('{}/image.png'.format(TRAIN_OUT_DIR), img[0].cpu().numpy())
            plt.imsave('{}/gt.png'.format(TRAIN_OUT_DIR),
                       np.moveaxis(gt[0].cpu().numpy(), 0, -1)[:, :, 0], cmap='gray')
            sample['to_seg'] = np.asarray(img[0].cpu())
            sample['features'] = all_feats
        features.append(feat.cpu())
    save_image_grid(preview, preview_path)
    return np.concatenate(features, axis=0), labels, masks, images, filenames, sample


def build_patchcore(device, im_size):
    patchcore = PatchCore(device)
    patchcore.load(
        device=device,
        layers_to_extract_from=None,
        featuresampler=IdentitySampler(),
        anomaly_scorer_num_nn=5,
        nn_method=FaissNN(True, 4),
        im_size=im_size,
    )
    return patchcore


def fit_or_load_patchcore(patchcore, features, save_path, load):
    os.makedirs(save_path, exist_ok=True)
    if load:
        print('Loading index....')
        patchcore.load_from_path(save_path, '')
    else:
        patchcore.fit(torch.from_numpy(features))
        patchcore.save_to_path(save_path, '')


def predict_anomaly(patchcore, features, batch_size=PREDICT_BATCH):
    print('Predicting...')
    scores = [None] * len(features)
    segmentations = [None] * len(features)
    for i in tqdm(range(0, len(features), batch_size)):
        batch = features[i:i + batch_size]
        sc, seg = patchcore.predict(torch.from_numpy(batch))
        scores[i:i + batch_size] = sc
        segmentations[i:i + batch_size] = seg
    return np.array(scores), np.array(segmentations)


def minmax_normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo)


def _sam_masks_for_prompt(predictor, segmentation, use_th):
    """Run SAM once for a single anomaly map and accumulate its 3 masks."""
    bbs, point, mask_prompt, th = get_prompt(segmentation)
    totals = [np.zeros_like(segmentation) for _ in range(NUM_SAM_MASKS)]
    if use_th:
        for bb in bbs:
            masks, _, _ = predictor.predict(box=bb, mask_input=mask_prompt)
            for k in range(NUM_SAM_MASKS):
                totals[k] += masks[k]
    else:
        masks, _, _ = predictor.predict(
            point_coords=np.asarray([point]), point_labels=np.asarray([1]))
        for k in range(NUM_SAM_MASKS):
            totals[k] += masks[k]
    return totals, point, th


def save_feature_and_prompt_visualizations(predictor, sample, all_feat_test, segmentations):
    save_feat = sample['features']
    to_seg = sample['to_seg']
    grid_size = int(np.sqrt(all_feat_test[0, 0].shape[0]))
    for k in range(0, len(save_feat), 2):
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        for i in range(grid_size):
            for j in range(grid_size):
                axs[i, j].imshow(save_feat[k][0, i * grid_size + j].cpu(), cmap='gray')
                axs[i, j].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig('{}/features_{}.png'.format(TRAIN_OUT_DIR, k + 1))
        plt.close(fig)

    predictor.set_image(to_seg)
    totals, _, th = _sam_masks_for_prompt(predictor, segmentations[SAMPLE_IDX], use_th=True)
    plt.imsave('{}/mask_prompt.png'.format(TRAIN_OUT_DIR),
               segmentations[SAMPLE_IDX], cmap='gray')
    plt.imsave('{}/segmentation_prompt.png'.format(TRAIN_OUT_DIR), th, cmap='gray')
    for k in range(NUM_SAM_MASKS):
        plt.imsave('{}/segmentation{}.png'.format(TRAIN_OUT_DIR, k), totals[k], cmap='gray')


def _save_sam_figure(out_dir, fn, im, point, gt, sg, totals, score):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(im)
    axes[0].scatter([point[0]], [point[1]], color='red', s=10)
    axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(np.moveaxis(gt.cpu().numpy(), 0, -1)[:, :, 0], cmap='gray')
    axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(sg, cmap='gray'); axes[2].set_title("Anomaly Map"); axes[2].axis("off")
    axes[3].imshow(totals[0], cmap='gray'); axes[3].set_title("Prediction 1"); axes[3].axis("off")
    axes[4].imshow(totals[2], cmap='gray'); axes[4].set_title("Prediction 3"); axes[4].axis("off")
    fig.suptitle('Score: {}'.format(score), fontsize=16)
    plt.savefig('{}/out_{}'.format(out_dir, os.path.basename(fn)), dpi=300, bbox_inches="tight")
    plt.close()


def run_sam_refinement(predictor, images, anomaly_labels, masks_gt, segmentations,
                       scores, filenames, save, save_class, use_th):
    """Compute SAM masks for every test image and (optionally) save a capped subset."""
    sam_masks = [[] for _ in range(NUM_SAM_MASKS)]
    prompts = []
    counter_normal = counter_abnormal = 0
    out_dir = 'results/{}'.format(save_class)
    if save:
        os.makedirs(out_dir, exist_ok=True)
    for im, la, gt, sg, sc, fn in zip(images, anomaly_labels, masks_gt,
                                      segmentations, scores, filenames):
        predictor.set_image(im)
        totals, point, th = _sam_masks_for_prompt(predictor, sg, use_th)
        prompts.append(th)
        for k in range(NUM_SAM_MASKS):
            sam_masks[k].append(np.clip(totals[k], 0, 1))
        if save and ((la and counter_abnormal < MAX_SAVED_ABNORMAL)
                     or (not la and counter_normal < MAX_SAVED_NORMAL)):
            _save_sam_figure(out_dir, fn, im, point, gt, sg, totals, sc)
            if la:
                counter_abnormal += 1
            else:
                counter_normal += 1
    return sam_masks, prompts


def mean_dice(predictions, masks_gt, anomaly_labels):
    total, count = 0.0, 0
    for i, is_anomaly in enumerate(anomaly_labels):
        if is_anomaly:
            total += metric.binary.dc(predictions[i], masks_gt[i].cpu().detach().numpy())
            count += 1
    return total / count, count


def evaluate_pixel(segmentations, masks_gt, anomaly_labels):
    full = compute_pixelwise_retrieval_metrics(segmentations, masks_gt)['auroc']
    print('Pixel Auroc: {}'.format(full))
    sel = [i for i, a in enumerate(anomaly_labels) if a]
    pro = compute_pixelwise_retrieval_metrics(
        [segmentations[i] for i in sel],
        [masks_gt[i] for i in sel],
    )['auroc']
    print('Pro Auroc: {}'.format(pro))


def evaluate_dice(prompts, sam_masks, masks_gt, anomaly_labels):
    print(masks_gt[0].shape)
    dice_th, n = mean_dice(prompts, masks_gt, anomaly_labels)
    print('Dice on th {} = {}'.format(n, dice_th))
    for k, masks in enumerate(sam_masks):
        dice, _ = mean_dice(masks, masks_gt, anomaly_labels)
        print('Dice SAM {} = {}'.format(k, dice))


def main(args):
    os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
    predictor = get_sam_model("vit_b", args.device, args.med)
    train_dl, test_dl, use_mask = load_data(args.device, args.dataset, args.size)

    train_features = None
    if not args.load:
        train_features = extract_train_features(
            predictor, train_dl, '{}/train_set.png'.format(TRAIN_OUT_DIR))

    patchcore = build_patchcore(args.device, args.size)
    save_path = os.path.join('checkpoints/nn_indexes', args.dataset)
    fit_or_load_patchcore(patchcore, train_features, save_path, args.load)
    del train_features
    torch.cuda.empty_cache()

    if not use_mask:
        print('Impossible to compute pixelwise metrics for dataset "{}", missing GT masks'
              .format(args.dataset))
        return

    all_feat_test, labels_gt, masks_gt, images_gt, filenames, sample = \
        extract_test_features(predictor, test_dl,
                              '{}/test_set.png'.format(TRAIN_OUT_DIR), args.save)

    scores, segmentations = predict_anomaly(patchcore, all_feat_test)
    scores = minmax_normalize(scores)
    segmentations = minmax_normalize(segmentations)
    anomaly_labels = [x != 0 for x in labels_gt]

    auroc = compute_imagewise_retrieval_metrics(scores, anomaly_labels)['auroc']
    print('Auroc: {}'.format(auroc))
    evaluate_pixel(segmentations, masks_gt, anomaly_labels)

    if args.save and sample['features'] is not None:
        save_feature_and_prompt_visualizations(predictor, sample, all_feat_test, segmentations)

    sam_masks, prompts = run_sam_refinement(
        predictor, images_gt, anomaly_labels, masks_gt, segmentations, scores,
        filenames, args.save, args.dataset, args.use_th)
    evaluate_dice(prompts, sam_masks, masks_gt, anomaly_labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="The device to run on.")
    parser.add_argument("--dataset", type=str, default="RESC",
                        choices=['RESC', 'BRAIN', 'LIVER'], help="Data type.")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--med", action="store_true")
    parser.add_argument("--use_th", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('{} {}'.format('Evaluating' if args.load else 'Training', args.dataset))
    main(args)
