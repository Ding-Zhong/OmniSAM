import torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from utils.metrics import fast_hist, per_class_iu
from utils.misc import AverageMeter
from utils.io import save_state_dict
from adaptation.strategies import BaseMemStrategy
from adaptation.methods import PrototypicalAdaptation, PrototypicalAdaptationEMA


class Trainer:
    def __init__(self, args, model_tgt, model_src, optimizer, criteria: Dict[str, torch.nn.Module],
                 adaptation_kwargs: Dict, mem_strategy: BaseMemStrategy,
                 callbacks: List, device):
        self.args = args
        self.model_tgt = model_tgt
        self.model_src = model_src
        self.optimizer = optimizer
        self.criteria = criteria
        self.device = device
        self.callbacks = callbacks
        self.mem_strategy = mem_strategy

        self.proto_bank = PrototypicalAdaptationEMA(**adaptation_kwargs)
        self.alpha = self.args.alpha

    def train_one_epoch(self, epoch: int, global_step, source_loader, target_loader, is_warmup):
        self.model_tgt.train()
        self.model_src.eval()

        src_it = iter(source_loader)
        tgt_it = iter(target_loader)

        for it in tqdm(range(len(target_loader)), desc=f"Epoch {epoch+1} [Train]"):
            batch_s = next(src_it)
            batch_t = next(tgt_it)

            self.optimizer.zero_grad()

            global_step = self.mem_strategy.forward(global_step, self.model_tgt, self.model_src, batch_s, batch_t,
                                               self.criteria, self.proto_bank, self.alpha, self.device, self.optimizer, self.callbacks, is_warmup)
        
        return global_step

    def validate(self, epoch: int, val_loader, class_names: List[str]):
        self.model_tgt.eval()
        hist = np.zeros((len(class_names), len(class_names)), dtype=np.float64)

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        with torch.no_grad():
            for idx, batch in enumerate(progress_bar):
                images, labels, *_ = batch
                images, labels = images.to(self.device), labels.to(self.device)
                frame_cnt = images.shape[1]
                for frame_idx in range(frame_cnt):
                    preds = self.mem_strategy.forward_eval(self.model_tgt, images, frame_idx)
                    pred = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy()
                    label_np = labels[:, frame_idx].squeeze(0).cpu().numpy()
                    hist += fast_hist(label_np.flatten(), pred.flatten(), len(class_names))
                self.mem_strategy.reset_memory(self.model_tgt)

        mIoUs = per_class_iu(hist)
        for i, cname in enumerate(class_names):
            print(f"{cname:<15}: {round(mIoUs[i]*100, 2)}")
        cur_mIoU = round(np.nanmean(mIoUs)*100, 2)

        for cb in self.callbacks:
            if hasattr(cb, 'on_validation_end'):
                cb.on_validation_end(epoch, cur_mIoU, self.model_tgt)

        return cur_mIoU

    def reset_optimizer(self, lr: float):
        self.optimizer.state.clear()