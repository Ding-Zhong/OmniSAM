from typing import Protocol, Tuple
import torch
import numpy as np
import wandb

class BaseMemStrategy(Protocol):
    def forward(self, model, batch, criterion, device) -> Tuple[torch.Tensor,...]:
        ...
    def forward_eval(self, model, images, frame_idx): ...
    def reset_memory(self, model): ...

class WithMemStrategy:
    def forward(self, 
                global_step, 
                model,
                batch,
                criterion, 
                device, 
                optimizer, 
                callbacks,
                rank
                ):
        images, labels, *_ = batch
        images, labels = images.to(device), labels.to(device)

        frame_cnt = images.shape[1]
        loss_seq = 0
        random_enable_mem = np.random.choice([1, 0], p=[0.8, 0.2])

        for frame_idx in range(frame_cnt):
            frame_img = images[:, frame_idx, :, :, :]  # [B, C, H, W]
            frame_lbl = labels[:, frame_idx, :, :]      # [B, H, W]

            if random_enable_mem:
                pred = model(frame_idx, frame_img)
            else:
                pred= model(0, frame_img)

            loss = criterion(pred, frame_lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_seq += loss.item()
    
        loss_avg = loss_seq / frame_cnt

        for cb in callbacks:
            if hasattr(cb, 'on_iter_end'):
                cb.on_iter_end(global_step=global_step,
                            loss=loss_avg,
                            lr=optimizer.param_groups[0]['lr'],
                            rank=rank) 

        self.reset_memory(model)


    def forward_eval(self, model, images, frame_idx):
        return model(frame_idx, images[:, frame_idx], mem_type='source')

    def reset_memory(self, model):
        model.module.memory_bank._init_output_dict_source()
        model.module.memory_bank._init_output_dict_target()


class NoMemStrategy(WithMemStrategy):
    def forward(self, 
                global_step, 
                model,
                batch,
                criterion,
                device,
                optimizer, 
                callbacks,
                rank
                ):
        images, labels, *_ = batch
        images, labels = images.to(device), labels.to(device)

        pred = model(0, images)
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()


        for cb in callbacks:
            if hasattr(cb, 'on_iter_end'):
                cb.on_iter_end(global_step=global_step,
                            loss=loss.item(),
                            lr=optimizer.param_groups[0]['lr'],
                            rank=rank) 

        self.reset_memory(model)

    def forward_eval(self, model, images, frame_idx):
        return model(0, images[:, frame_idx], mem_type='source')