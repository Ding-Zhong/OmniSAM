import os
from typing import Optional
import torch

from utils.build import *
from adaptation.pl_generator import PseudoLabelGenerator
from adaptation.trainer import Trainer
from adaptation.strategies import WithMemStrategy, NoMemStrategy
from utils.callbacks import CheckpointCB, LRCB, LoggerCB
from utils.io import ensure_dir, clear_dir
from utils.lr_scheduler import build_poly_warmup_scheduler

class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_builder = None
        self.trainer = None

        self.class_names = None
        self.num_classes = None
        self.is_warmup = None

        self.target_pl_loader = None
        self.source_loader = None
        self.val_loader = None
        self.target_loader = None

        self.model_t = None
        self.model_s = None
        self.save_dir = None

        self.global_step = 0
        self.best_mIoU = 0.0
        self.best_ckpt_path: Optional[str] = None

    def setup(self):
        self.data_builder = DatasetBuilder(self.args)
        self.target_pl_loader, self.class_names, self.num_classes, frame_cnt = self.data_builder.build_tgt_for_pl()

        self.model_t = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)
        self.model_s = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)

        load_checkpoint(self.model_t, self.args.tgt_ckpt)
        load_checkpoint(self.model_s, self.args.src_ckpt)

        criteria = self.data_builder.build_criteria(self.device, self.args.dataset)
        optimizer = torch.optim.AdamW(self.model_t.parameters(), lr=self.args.lr, weight_decay=1e-4)

        scheduler = build_poly_warmup_scheduler(optimizer, power=1.0, warmup_iters=1800)

        mem_strategy = WithMemStrategy() if self.args.dataset != 'Stanford2D3D' else NoMemStrategy()

        callbacks = [
            CheckpointCB(save_dir=self._save_dir(), backbone=self.args.backbone),
            LRCB(scheduler),
            LoggerCB()
        ]

        if self.args.dataset == 'Stanford2D3D':
            temp_path_file = os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)
        else:
            temp_path_file = os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, self.args.dataset, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)
        
        self.trainer = Trainer(args=self.args,
                               model_tgt=self.model_t,
                               model_src=self.model_s,
                               optimizer=optimizer,
                               criteria=criteria,
                               adaptation_kwargs=dict(num_classes=self.num_classes,
                                                    feature_dim=256),
                               mem_strategy=mem_strategy,
                               callbacks=callbacks,
                               device=self.device)

    def need_pseudo_labels(self, epoch: int) -> bool:
        return (epoch % self.args.pseudo_every) == 0

    def generate_pseudo_labels(self, epoch: int):
        self.save_dir = self._pl_epoch_dir(epoch)
        parent_save_dir = os.path.dirname(self.save_dir)
        clear_dir(self.save_dir)
        max_num = self.args.warmup_pl if epoch == 0 else self.args.iters_pl
        self.is_warmup = True if epoch == 0 else False

        plg = PseudoLabelGenerator(model=self.model_t,
                                   device=self.device,
                                   class_names=self.class_names,
                                   num_classes=self.num_classes)
        used_paths = plg.run(self.target_pl_loader, self.save_dir, max_num, self.args.dataset, self.args.uc_threshold)

        self.data_builder.write_splits_after_pl(parent_save_dir, used_paths)

    def train(self, epoch: int):
        if self.save_dir is None:
            self.save_dir = self._pl_epoch_dir(epoch) 
        self.target_loader, self.val_loader = self.data_builder.build_tgt_from_pd(self.save_dir)
        if self.source_loader is None:
            self.source_loader, _ = self.data_builder.build_src()
        self.global_step = self.trainer.train_one_epoch(epoch, self.global_step, self.source_loader, self.target_loader, self.is_warmup)

    def validate(self, epoch: int):
        mIoU = self.trainer.validate(epoch, self.val_loader, self.class_names)
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            self.best_ckpt_path = os.path.join(self._save_dir(), f"best_{self.args.backbone}_iou{mIoU:.2f}.pth")
            torch.save(self.model_t.state_dict(), self.best_ckpt_path)
        print(f"Epoch {epoch+1}: val mIoU={mIoU:.2f}, best={self.best_mIoU:.2f}")

    def step_epoch_end(self, epoch: int):
        if (epoch + 1) % 5 == 0:
            self.trainer.reset_optimizer(lr=self.args.lr)

    def _save_dir(self):
        d = os.path.join(self.args.pd_root, 'checkpoints', self.args.dataset, self.args.backbone)
        ensure_dir(d)
        return d

    def _pl_epoch_dir(self, epoch: int):
        if self.args.dataset == 'Stanford2D3D':
            return os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, f'epoch{epoch + 1}')
        else:
            return os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, self.args.dataset,
                                f'epoch{epoch + 1}')