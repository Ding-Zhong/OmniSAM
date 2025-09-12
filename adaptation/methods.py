import torch
import torch.nn.functional as F

class PrototypicalAdaptation:
    def __init__(self, num_classes, feature_dim, frame_cnt):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.frame_cnt = frame_cnt
        # Initialize global prototypes for each class
        self.global_prototypes = torch.zeros((frame_cnt, num_classes, feature_dim), requires_grad=False).to('cuda')

    def update_global_prototypes(self, global_step, frame_idx, current_prototypes):
        self.global_prototypes[frame_idx] = (1 - 1 / global_step) * self.global_prototypes[frame_idx] + (1 / global_step) * current_prototypes.detach()

    def calculate_batch_prototypes(self, features, labels):
        batch_prototypes = torch.zeros((self.num_classes, self.feature_dim), device=features.device)
        count = torch.zeros(self.num_classes, device=features.device)
        
        labels = labels.to(features.device)
        labels = labels.unsqueeze(1)

        features = F.interpolate(features, size=(256, 256), mode='bilinear', align_corners=False)
        labels_resized = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long().squeeze(1)

        b, c, h, w = features.size()
        features = features.permute(0, 2, 3, 1).reshape(-1, c)
        labels_resized = labels_resized.view(-1)

        for i in range(self.num_classes):
            mask = (labels_resized == i)
            if mask.sum() > 0:
                batch_prototypes[i] = features[mask].mean(dim=0)

        return batch_prototypes

    def prototype_loss(self, frame_idx, batch_prototypes):
        # Calculate the loss between batch prototypes and global prototypes
        loss = F.mse_loss(batch_prototypes, self.global_prototypes[frame_idx])

        return loss

class PrototypicalAdaptationEMA:
    def __init__(self, num_classes, feature_dim, momentum=0.99, device="cuda", ignore_index=255, conf_thres=0.85):
        self.num_classes = num_classes
        self.feat_dim = feature_dim
        self.momentum = momentum
        self.ignore_index = ignore_index
        self.conf_thres = conf_thres

        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.counts     = torch.zeros(num_classes, device=device)

    def snapshot(self):
        self._proto_prev = self.prototypes.detach().clone()

    def delta_mean(self):
        if not hasattr(self, "_proto_prev"):
            return 0.0
        return (self.prototypes - self._proto_prev).abs().mean().item()
        
    @torch.no_grad()
    def update(self, feats, labels, conf=None):
        if feats.dim() == 4:
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
            if labels.dim() == 3:      # (B,H,W)
                labels = labels.unsqueeze(1)       # -> (B,1,H,W)
            if labels.shape[-2:] != (H, W):
                labels = F.interpolate(labels.float(), size=(H, W), mode='nearest').long()
            labels = labels.view(-1)

        mask = labels != self.ignore_index
        if conf is not None:
            mask &= conf > self.conf_thres  # 举例

        if mask.sum() == 0:
            return

        feats = F.normalize(feats[mask], dim=1)
        labels = labels[mask]

        sums = torch.zeros(self.num_classes, self.feat_dim, device=feats.device)
        sums.index_add_(0, labels, feats)

        cnts = torch.bincount(labels, minlength=self.num_classes).to(feats.device)

        valid = cnts > 0

        m = self.momentum
        self.prototypes[valid] = m * self.prototypes[valid] + (1 - m) * (sums[valid] / cnts[valid].unsqueeze(1))
        self.counts[valid] += cnts[valid]

    def mse_pull_loss(self, feats, labels):
        if feats.dim() == 4:
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
            if labels.dim() == 3:      # (B,H,W)
                labels = labels.unsqueeze(1)       # -> (B,1,H,W)
            if labels.shape[-2:] != (H, W):
                labels = F.interpolate(labels.float(), size=(H, W), mode='nearest').long()
            labels = labels.view(-1)

        mask = labels != self.ignore_index
        if mask.sum() == 0:
            return torch.tensor(0., device=feats.device)

        feats = feats[mask]
        labels = labels[mask]

        sums = torch.zeros(self.num_classes, self.feat_dim, device=feats.device)
        sums.index_add_(0, labels, feats)
        cnts = torch.bincount(labels, minlength=self.num_classes).to(feats.device)
        valid = cnts > 0
        batch_proto = torch.zeros_like(self.prototypes)
        batch_proto[valid] = sums[valid] / cnts[valid].unsqueeze(1)

        return F.mse_loss(batch_proto[valid], self.prototypes[valid])