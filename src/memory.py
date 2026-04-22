"""PatchCore and PatchCore detection methods."""
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from src.common import (
    Aggregator,
    FaissNN,
    IdentitySampler,
    NearestNeighbourScorer,
    Preprocessing,
    RescaleSegmentor,
)

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        layers_to_extract_from,
        device,
        anomaly_score_num_nn=1,
        featuresampler=IdentitySampler(),
        nn_method=FaissNN(False, 4),
        im_size=256,
        **kwargs,
    ):
        self.device = device
        self.layers_to_extract_from = layers_to_extract_from
        self.forward_modules = torch.nn.ModuleDict({})
        self.patch_maker = PatchMaker(5, stride=2)#TODO: adjust       5, 2  

        feature_dimensions = [(256,64,64)]
        self.preprocessing = Preprocessing(
            feature_dimensions, 1024
        )

        self.target_embed_dimension = 1024#384
        self.preadapt_aggregator = Aggregator(
            target_dim=self.target_embed_dimension
        )

        _ = self.preadapt_aggregator.to(self.device)

        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=im_size
        )

        self.featuresampler = featuresampler
    
    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, features, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features
         
        if provide_patch_shapes:
            if self.layers_to_extract_from :
                features = np.squeeze(features, axis=2)
                features = [features[layer] for layer in self.layers_to_extract_from]
            else:
                features = [features]
        else:
            if self.layers_to_extract_from :
                features = [features[layer] for layer in self.layers_to_extract_from]
            else:
                features = [features.unsqueeze(0)]
        #if provide_patch_shapes:
        #print('Features before patch', len(features), features[0].shape)
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        #print('Patched features', len(features), features[0][0].shape) # SINGLE:1 torch.Size([1, 1024, 256, 3, 3]) | MULTI: 3 torch.Size([1, 1024, 256, 3, 3])
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        #print(features[0].shape) # 16384, 64, 3, 3
        #print('Features before pre', len(features), features[0].shape)
        features = self.preprocessing(features) 
        #print('Features after pre', len(features), features.shape)
        features = self.preadapt_aggregator(features)
        #print('Features after agg', len(features), features.shape)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    
    def fit(self, training_data, return_patch=False):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        if return_patch:
            patches = self._fill_memory_bank(training_data, return_patch)
            return patches
        else:
            self._fill_memory_bank(training_data, return_patch)
            return None

    def _fill_memory_bank(self, images, return_patch):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image#.to(torch.float).to(self.device)
                emb = self._embed(input_image)
                return emb
        features = []  
        for image in tqdm.tqdm(images, desc="Patching..."):
            features.append(_image_to_features(image))
        del images
        torch.cuda.empty_cache()

        features = np.concatenate(features, axis=0)
        #print('Before')
        features = self.featuresampler.run(features)
        #print('Detection_feat', features.shape)
        #print('After')
        if return_patch:
            return features
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def predict_feat(self, features):
        with torch.no_grad():
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            #print(image_scores.shape)
            #print(image_scores[0])
            imagescores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        #images = images.to(torch.float).to(self.device)
        #_ = self.forward_modules.eval()
        if self.layers_to_extract_from:
            batchsize = images.shape[1]
        else:
            batchsize = images.shape[0]
        #batchsize = len(images)
        #print('Batch size: {}'.format(batchsize))
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            #print(features.shape)#(5120, 3072)   
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            #print(patch_scores.shape)#(5120,)
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            #print(image_scores.shape)#(5, 1024)  
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            #print(image_scores.shape)#(5, 1024, 1)
            image_scores = self.patch_maker.score(image_scores)
            #print(image_scores.shape)#(5,) --> Uno score per immagine del batch

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            #print(patch_scores.shape)#(5, 1024)
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            #print(patch_scores.shape)#(5, 32, 32) --> 5 immagini con ?x?

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            #print(len(masks), masks[0].shape)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
