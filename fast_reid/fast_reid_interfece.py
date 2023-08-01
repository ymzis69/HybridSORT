import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# from torch.backends import cudnn

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer
from fast_reid.fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch

# cudnn.benchmark = True


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False

    cfg.freeze()

    return cfg


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


def preprocess(image, input_size):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[1], input_size[0], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size) * 114
    img = np.array(image)
    r = min(input_size[1] / img.shape[0], input_size[0] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    return padded_img, r


class FastReIDInterface:
    def __init__(self, config_file, weights_path, device, batch_size=16):
        super(FastReIDInterface, self).__init__()
        if device != 'cpu':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.batch_size = batch_size    # 8

        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])

        self.model = build_model(self.cfg)
        self.model.eval()

        Checkpointer(self.model).load(weights_path)

        if self.device != 'cpu':
            self.model = self.model.eval().to(device='cuda').half()
        else:
            self.model = self.model.eval()

        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST     # [384, 128]

    def inference(self, image, detections):

        if detections is None or np.size(detections) == 0:
            return []

        H, W, _ = np.shape(image)       # original size, [1080, 1920] for MOT17

        batch_patches = []
        patches = []
        for d in range(np.size(detections, 0)):     # iteration over detections
            tlbr = detections[d, :4].astype(np.int_)
            tlbr[0] = max(0, tlbr[0])           # clamp
            tlbr[1] = max(0, tlbr[1])           # clamp
            tlbr[2] = min(W - 1, tlbr[2])       # clamp
            tlbr[3] = min(H - 1, tlbr[3])       # clamp
            patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]      # crop image, BGR

            # the model expects RGB inputs
            patch = patch[:, :, ::-1]

            # Apply pre-processing to image.
            patch = cv2.resize(patch, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_LINEAR)    # [384, 128, 3]
            # patch, scale = preprocess(patch, self.cfg.INPUT.SIZE_TEST[::-1])

            # plt.figure()
            # plt.imshow(patch)
            # plt.show()

            # Make shape with a new batch dimension which is adapted for network input
            patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))     # [3, 384, 128]
            patch = patch.to(device=self.device).half()

            patches.append(patch)

            if (d + 1) % self.batch_size == 0:      # if already get a batch
                patches = torch.stack(patches, dim=0)
                batch_patches.append(patches)
                patches = []

        if len(patches):        # stack each batch
            patches = torch.stack(patches, dim=0)
            batch_patches.append(patches)

        features = np.zeros((0, 2048))          # TODO: [hgx1001] need to be set by hand
        # features = np.zeros((0, 768))

        for patches in batch_patches:       # iteration over batch
            # Run model
            patches_ = torch.clone(patches)     # [8, 3, 384, 128]
            pred = self.model(patches)          # [8, 2048]
            pred[torch.isinf(pred)] = 1.0

            feat = postprocess(pred)            # normalization() and numpy()

            nans = np.isnan(np.sum(feat, axis=1))
            if np.isnan(feat).any():        # handle nans, pass for now
                for n in range(np.size(nans)):
                    if nans[n]:
                        # patch_np = patches[n, ...].squeeze().transpose(1, 2, 0).cpu().numpy()
                        patch_np = patches_[n, ...]
                        patch_np_ = torch.unsqueeze(patch_np, 0)
                        pred_ = self.model(patch_np_)

                        patch_np = torch.squeeze(patch_np).cpu()
                        patch_np = torch.permute(patch_np, (1, 2, 0)).int()
                        patch_np = patch_np.numpy()

                        plt.figure()
                        plt.imshow(patch_np)
                        plt.show()

            features = np.vstack((features, feat))

        return features     # [n_det, 2048]

