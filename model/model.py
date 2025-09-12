import torch
import torch.nn as nn
import torch.nn.functional as F
from sam.sam_lora_image_encoder import LoRA_Sam
from model.modules.decoder_sam import SegFormerHead, SegFormerHead_adaption
from model import MemoryBank

class OmniSAM(nn.Module):
    def __init__(self, sam_model, num_classes, num_maskmem):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        self.decoder = SegFormerHead(num_classes=num_classes)
        # self.decoder = DMLPHead(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)

    def forward(self, frame_idx, x, mem_type="source"):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        x = backbone_out['backbone_fpn']
        x[-1] = self.memory_bank(
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds, 
                    num_obj_ptr_tokens=0,
                    mem_type=mem_type
                )
        x = self.decoder(x)
        pred = F.interpolate(x, size=(height,width), mode='bilinear', align_corners=False)
        # Encode new memory
        if self.num_maskmem > 0: 
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type)

        return pred

class OmniSAM_adapt(nn.Module):
    def __init__(self, sam_model, num_classes=13, num_maskmem=9):
        super().__init__()
        self.sam_model = sam_model
        self.encoder = LoRA_Sam(sam_model, 4)
        self.decoder = SegFormerHead_adaption(num_classes=num_classes)
        # self.decoder = DMLPHead_adaption(num_classes=num_classes)
        self.num_maskmem = num_maskmem
        self.memory_bank = MemoryBank(sam_model, num_maskmem=num_maskmem)

    def forward(self, frame_idx, x, mem_type="source"):
        _, _, height, width = x.shape
        backbone_out = self.encoder(x)
        current_vision_feats = backbone_out['vision_features']
        current_vision_pos_embeds = backbone_out['vision_pos_enc'][-1]
        feat = backbone_out['backbone_fpn']
        feat[-1] = self.memory_bank(
                    frame_idx,
                    current_vision_feats,
                    current_vision_pos_embeds, 
                    mem_type=mem_type,
                    num_obj_ptr_tokens=0
                )
        x, feat_de = self.decoder(feat)
        pred = F.interpolate(x, size=(height,width), mode='bilinear', align_corners=False)
        # Encode new memory
        if self.num_maskmem > 0: 
            output = torch.argmax(pred, 1)
            mask_for_mem = output.unsqueeze(1)
            self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type)
        
        return pred, feat, feat_de