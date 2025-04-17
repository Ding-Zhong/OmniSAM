import torch
import torch.nn.functional as F

class PrototypicalAdaptation:
    def __init__(self, num_classes, feature_dim, num_maskmem):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_maskmem = num_maskmem
        # Initialize global prototypes for each class
        self.global_prototypes = torch.zeros((num_maskmem, num_classes, feature_dim), requires_grad=False).to('cuda')

    def update_global_prototypes(self, global_step, frame_idx, current_prototypes):
        self.global_prototypes[frame_idx] = (1 - 1 / global_step) * self.global_prototypes[frame_idx] + (1 / global_step) * current_prototypes.detach()

    def compute_loss(self, features, labels):
        batch_prototypes = self.calculate_batch_prototypes(features, labels)
        self.update_global_prototypes(batch_prototypes)
        prototype_loss = self.prototype_loss(batch_prototypes)
        total_loss = prototype_loss

        return total_loss

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
