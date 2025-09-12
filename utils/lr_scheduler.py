from torch.optim.lr_scheduler import LambdaLR


def build_poly_warmup_scheduler(optimizer, power=1.0, warmup_iters=100, total_iters=20000):
    def lr_lambda(cur_iter):
        if cur_iter < warmup_iters:
            return float(cur_iter) / float(max(1, warmup_iters))
        else:
            return (1 - (cur_iter - warmup_iters) / float(max(1, total_iters - warmup_iters))) ** power
    return LambdaLR(optimizer, lr_lambda)