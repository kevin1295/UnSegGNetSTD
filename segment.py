from bilateral_solver import bilateral_solver_output
from features_extract import deep_features
from torch_geometric.data import Data
from extractor import ViTExtractor
import torch.optim as optim
import numpy as np
import torch
import util
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from gnn_pool import GNNpool 
from transformers import SamModel, SamProcessor

class Segmentation:
    def __init__(self, process, bs=False, epochs=20, resolution=(224, 224), device=None,
                 activation=None, loss_type=None, threshold=None, conv_type=None,
                 alpha=2.0, lam=1.0, eps=3.0, nb_iter=3, gen_init=False,
                 checkpoint_path='model.pt'):
        if process not in ["KMEANS_DINO", "DINO", "MEDSAM_INFERENCE"]:
            raise ValueError(f'Process: {process} is not supported')
        self.process = process
        self.resolution = resolution
        self.bs = bs
        self.epochs = epochs
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_type = loss_type
        self.threshold = threshold
        self.activation = activation
        self.checkpoint_path = Path(checkpoint_path)
        if process in ["DINO", "KMEANS_DINO"]:
            self.feats_dim = 384
            pretrained_weights = Path(__file__).resolve().parent / 'dino_deitsmall8_pretrain_full_checkpoint.pth'
            if not pretrained_weights.exists():
                url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth'
                util.download_url(url, pretrained_weights)
            self.extractor = ViTExtractor(model_dir=pretrained_weights, device=self.device)
            self.model = GNNpool(self.feats_dim, 64, 32, 2,
                                 self.device, activation, loss_type, conv_type,
                                 alpha=alpha, lam=lam, entropy_epsilon=eps, nb_iterations=nb_iter
                                 ).to(self.device)
        elif process == "MEDSAM_INFERENCE":
            self.processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
            self.model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(self.device)
        if gen_init:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
        elif process not in ["KMEANS_DINO", "MEDSAM_INFERENCE"] and not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f'Initialization checkpoint not found: {self.checkpoint_path}. '
                'Run with --gen_init once or provide --checkpoint.'
            )
        self.model.train()

    def segment(self, image, mask, prompt_box=None):
        """
        @param image: Image to segment (numpy array)
        @param mask: Ground truth mask (binary numpy array)
        @param prompt_box: Optional box prompt for MEDSAM inference
        """
        if self.process == "MEDSAM_INFERENCE":
            if prompt_box is None:
                raise ValueError('MEDSAM_INFERENCE requires an explicit prompt_box; ground truth masks are not used as prompts.')
            medsam_seg = self.medsam_inference(image, prompt_box)
            segmentation = cv2.resize(medsam_seg.astype('float'), (image[:, :, 0].shape[1], image[:, :, 0].shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            image_tensor, image = util.load_data_img(image, self.resolution)
            if self.process in ["DINO", "KMEANS_DINO"]:
                F = deep_features(image_tensor, self.extractor, device=self.device)

            if self.process == "KMEANS_DINO":
                kmeans = KMeans(n_clusters=2, random_state = 42)
                kmeans.fit(F)
                labels = kmeans.labels_
                S = torch.tensor(labels)
            else:
                W = util.create_adj(F, self.loss_type, self.threshold)
                node_feats, edge_index, edge_weight = util.load_data(W, F)
                data = Data(node_feats, edge_index, edge_weight).to(self.device)
                # 加载模型权重
                model_weights = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                current_state = self.model.state_dict()
                shared_weights = {
                    key: value for key, value in model_weights.items()
                    if key in current_state and current_state[key].shape == value.shape
                }
                load_result = self.model.load_state_dict(shared_weights, strict=False)
                allowed_missing = {'alpha', 'lam', 'entropy_epsilon'}
                unexpected_missing = set(load_result.missing_keys) - allowed_missing
                if unexpected_missing:
                    raise RuntimeError(
                        f'Checkpoint is missing shared model parameters: {sorted(unexpected_missing)}'
                    )
                opt = optim.AdamW(self.model.parameters(), lr=0.001)

                for _ in range(self.epochs):
                    opt.zero_grad()
                    A, S = self.model(data, torch.from_numpy(W).to(self.device))
                    loss = self.model.loss(A, S)
                    loss.backward()
                    opt.step()

                S = S.detach().cpu()
                S = torch.argmax(S, dim=-1)
                
            segmentation = util.graph_to_mask(S, image_tensor, image)

        if self.bs:
            segmentation = bilateral_solver_output(image, segmentation)[1]
        
        segmentation = np.where(segmentation==True, 1,0).astype(np.uint8)  
        
        iou1 = Segmentation.iou(segmentation, mask)
        iou2 = Segmentation.iou(1-segmentation, mask)
        if iou2 > iou1:
            segmentation = 1 - segmentation

        segmentation_over_image  = util.apply_seg_map(image, segmentation, 0.1)

        return max(iou1, iou2), segmentation, segmentation_over_image

    def medsam_inference(self, image, input_box):
        """
        @param image: Image to segment (numpy array)
        @param input_box: Box prompt in [x_min, y_min, x_max, y_max] format
        """
        image = cv2.resize(image.astype('float'), self.resolution, interpolation=cv2.INTER_NEAREST)
        inputs = self.processor(image, input_boxes=[[[input_box]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        
        medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

        return medsam_seg

    @staticmethod
    def iou(mask1, mask2):
        """
        Calculate Intersection over Union between two masks
        @param mask1: Binary numpy array
        @param mask2: Binary numpy array
        """
        x = mask1.ravel()
        y = mask2.ravel()
        intersection = np.logical_and(x, y)
        union = np.logical_or(x, y)
        union_size = np.sum(union)
        if union_size == 0:
            return 1.0
        similarity = np.sum(intersection) / union_size
        return similarity
