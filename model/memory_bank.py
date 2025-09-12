import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

class MemoryBank(nn.Module):
    def __init__(self, sam_model, num_maskmem=6):
        super().__init__()
        self.sam_model = sam_model
        self.mask_encoder = sam_model.memory_encoder
        self.num_maskmem = num_maskmem
        self.hidden_dim = sam_model.image_encoder.neck.d_model
        self.mem_dim = sam_model.memory_encoder.out_proj.weight.shape[0]
        self.output_dict_source = {"frame_outputs": {}}
        self.output_dict_target = {"frame_outputs": {}}
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_pos_enc, std=0.02)

    def _init_output_dict_source(self):
        self.output_dict_source = {"frame_outputs": {}}

    def _init_output_dict_target(self):
        self.output_dict_target = {"frame_outputs": {}}

    def _encode_new_memory(self, frame_idx, current_vision_feats, mask_for_mem, mem_type):
        maskmem_out = self.sam_model.memory_encoder(current_vision_feats, mask_for_mem)
        current_memory_feats = maskmem_out['vision_features']
        current_memory_pos_embeds = maskmem_out['vision_pos_enc']
        current_memory_feats = current_memory_feats.flatten(2).permute(2, 0, 1)
        current_memory_pos_embeds = current_memory_pos_embeds[-1].flatten(2).permute(2, 0, 1)
        if mem_type == "source":
            self.output_dict_source["frame_outputs"][frame_idx] = {"maskmem_features": current_memory_feats.detach(), "maskmem_pos_enc": current_memory_pos_embeds.detach()}
        elif mem_type == "target":
            self.output_dict_target["frame_outputs"][frame_idx] = {"maskmem_features": current_memory_feats.detach(), "maskmem_pos_enc": current_memory_pos_embeds.detach()}
        else:
            raise ValueError("Wrong memory bank type.")

    def forward(self, frame_idx, current_vision_feats, current_vision_pos_embeds, mem_type, num_obj_ptr_tokens=0):
        B, C, H, W = current_vision_feats.shape
        device = current_vision_feats.device
        vision_feats = current_vision_feats.flatten(2).permute(2, 0, 1)
        vision_pos_embeds = current_vision_pos_embeds.flatten(2).permute(2, 0, 1)
        
        if frame_idx != 0:
            to_cat_memory, to_cat_memory_pos_embed, t_pos_and_prevs = [], [], []
            for t_pos in range(self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                prev_frame_idx = frame_idx - t_rel

                if mem_type == "source":
                    out = self.output_dict_source["frame_outputs"].get(prev_frame_idx, None)
                elif mem_type == "target":
                    out = self.output_dict_target["frame_outputs"].get(prev_frame_idx, None)
                else:
                    raise ValueError("Wrong memory bank type.")
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats)
                maskmem_enc = prev["maskmem_pos_enc"].to(device)
                # maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - frame_idx - 1].to(device))
                maskmem_enc = (maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1].to(device))
                to_cat_memory_pos_embed.append(maskmem_enc)
        
        else:
            pix_feat_with_mem = vision_feats + self.no_mem_embed.to(device)
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem

        memory = torch.cat(to_cat_memory, dim=0).to(device)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0).to(device)
        pix_feat_with_mem = self.sam_model.memory_attention(
            curr=vision_feats,
            curr_pos=vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
