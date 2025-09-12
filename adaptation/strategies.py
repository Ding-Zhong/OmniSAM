from typing import Protocol, Tuple
import torch
import wandb

def linear_ramp(cur, start_step, end_step, max_w):
    if cur < start_step: return 0.0
    if cur > end_step:   return max_w
    return max_w * (cur - start_step) / (end_step - start_step)
    
class BaseMemStrategy(Protocol):
    def forward(self, model, model_src, batch_s, batch_t, criteria, proto_bank, alpha, device) -> Tuple[torch.Tensor,...]:
        ...
    def forward_eval(self, model, images, frame_idx): ...
    def reset_memory(self, model): ...

class WithMemStrategy:
    def forward(self, 
                global_step, 
                model, 
                model_src, 
                batch_s, 
                batch_t, 
                criteria, 
                proto_bank, 
                alpha, 
                device, 
                optimizer, 
                callbacks, 
                is_warmup
                ):
        images_s, labels_s, *_ = batch_s
        images_t, labels_t, *_ = batch_t
        images_s, labels_s = images_s.to(device), labels_s.to(device)
        images_t, labels_t = images_t.to(device), labels_t.to(device)

        frame_cnt = images_t.shape[1]
        total_steps = 1000
        conf_thr = 0.85

        for frame_idx in range(frame_cnt):
            frame_img_s = images_s[:, frame_idx]
            frame_lbl_s = labels_s[:, frame_idx]
            frame_img_t = images_t[:, frame_idx]
            frame_pl_t = labels_t[:, frame_idx]

            pred_s, feat_s, feat_s_de = model(frame_idx, frame_img_s, mem_type='source')
            _, feat_s_ori, feat_s_de_ori = model_src(frame_idx, frame_img_s, mem_type='source')
            pred_t, feat_t, feat_t_de = model(frame_idx, frame_img_t, mem_type='target')
            with torch.no_grad():
                probs = torch.softmax(pred_t, 1)
                conf_t, pseudo_t = probs.max(1)
                mask_t = conf_t > conf_thr  
                
                proto_bank.snapshot()
                proto_bank.update(feat_s_de_ori, frame_lbl_s)             

            loss_sup_s = criteria['s'](pred_s, frame_lbl_s)
            loss_unsup = criteria['t'](pred_t, frame_pl_t)

            loss_proto  = proto_bank.mse_pull_loss(feat_t_de, frame_pl_t)

            beta_scheduled  = linear_ramp(global_step, 0, 0.1*total_steps*frame_cnt, 1.0)
            alpha_scheduled = linear_ramp(global_step, 0.1*total_steps*frame_cnt, 0.2*total_steps*frame_cnt, alpha)

            feat_t_de.retain_grad()
            loss_frame = loss_sup_s + beta_scheduled * loss_unsup + alpha_scheduled * loss_proto

            loss_frame.backward()
            optimizer.step()

            with torch.no_grad():
                grad_proto = (feat_t_de.grad.norm().item()
                            if feat_t_de.grad is not None else 0.0)
                ratio_proto = (alpha_scheduled * loss_proto.item()) / (loss_frame.item() + 1e-8)
                proto_delta = proto_bank.delta_mean()
                cov_t = mask_t.float().mean().item()
                conf_mean = conf_t[mask_t].mean().item() if mask_t.any() else 0.0

            extra_metrics = dict(
                loss_sup_s=loss_sup_s.item(),
                loss_unsup=loss_unsup.item(),
                loss_proto=loss_proto.item(),
                beta=beta_scheduled,
                alpha=alpha_scheduled,
                ratio_proto=ratio_proto,
                grad_proto=grad_proto,
                proto_delta=proto_delta,
                cov_t=cov_t,
                conf_mean=conf_mean,
            )

            for cb in callbacks:
                if hasattr(cb, 'on_iter_end'):
                    cb.on_iter_end(global_step=global_step,
                                loss=loss_frame.item(),
                                lr=optimizer.param_groups[0]['lr'],
                                metrics=extra_metrics) 

            global_step += 1

        # reset mem
        self.reset_memory(model)
        self.reset_memory(model_src)

        return global_step


    def forward_eval(self, model, images, frame_idx):
        return model(frame_idx, images[:, frame_idx], mem_type='source')[0]

    def reset_memory(self, model):
        model.memory_bank._init_output_dict_source()
        model.memory_bank._init_output_dict_target()


class NoMemStrategy(WithMemStrategy):
    def forward(self, 
                global_step, 
                model, 
                model_src, 
                batch_s, 
                batch_t, 
                criteria, 
                proto_bank, 
                alpha, 
                device, 
                callbacks,
                is_warmup
                ):
        images_t, labels_t, *_ = batch_t
        images_t, labels_t = images_t.to(device), labels_t.to(device)
        frame_cnt = images_t.shapep[1]

        for frame_idx in range(frame_cnt):
            images_s, labels_s, *_ = batch_s 
            images_s, labels_s = images_s.to(device), labels_s.to(device)

            frame_img_s = images_s
            frame_lbl_s = labels_s
            frame_img_t = images_t[:, frame_idx]
            frame_pl_t = labels_t[:, frame_idx]

            pred_s, _, feat_s_de = model(0, frame_img_s, mem_type='source')
            _, feat_s_ori, feat_s_de_ori = model_src(0, frame_img_s, mem_type='source')
            pred_t, _, feat_t_de = model(0, frame_img_t, mem_type='target')

            s_pt = proto_bank.calculate_batch_prototypes(feat_s_de_ori, frame_lbl_s)
            proto_bank.update_global_prototypes(global_step+1, frame_idx, s_pt)
            t_pt = proto_bank.calculate_batch_prototypes(feat_t_de, frame_pl_t)
            loss_pt = proto_bank.prototype_loss(frame_idx, t_pt)

            loss_unsup = criteria['t'](pred_t, frame_pl_t)
            loss_sup_s = criteria['s'](pred_s, frame_lbl_s)

            loss_frame = loss_sup_s + loss_unsup + alpha * loss_pt

            for cb in callbacks:
                if hasattr(cb, 'on_iter_end'):
                    cb.on_iter_end(global_step=global_step,
                                   loss=loss_frame.item(), lr=optimizer.param_groups[0]['lr'])

            loss_frame.backward()
            optimizer.step()

        self.reset_memory(model)
        self.reset_memory(model_src)

        return global_step
