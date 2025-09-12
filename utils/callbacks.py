import os
import torch
import wandb

class CheckpointCB:
    def __init__(self, save_dir, backbone):
        self.save_dir = save_dir
        self.backbone = backbone

    def on_validation_end(self, epoch, metric, model, rank=0):
        if rank == 0:
            log_dict = {"mIoU": metric}
            wandb.log(log_dict, step=epoch)
            path_last = os.path.join(self.save_dir, f'last_{self.backbone}.pth')
            torch.save(model.state_dict(), path_last)

class LRCB:
    def __init__(self, scheduler):
        self.scheduler = scheduler
    def on_iter_end(self, global_step, loss, lr, rank=0, metrics=None, **kwargs):
        if self.scheduler is not None:
            self.scheduler.step()

class LoggerCB:
    def on_iter_end(self, global_step, loss, lr, rank=0, metrics=None, **kwargs):
        if rank == 0:
            log_dict = {"loss": loss, "lr": lr}
            if metrics is not None:
                log_dict.update(metrics)
            wandb.log(log_dict, step=global_step)