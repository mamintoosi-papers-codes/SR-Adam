import torch
import math
from torch.optim import Optimizer

# ============================================================
# 1. SGD (Baseline)
# ============================================================

class SGDManual(Optimizer):
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                p.add_(grad, alpha=-group['lr'])
        return loss


# ============================================================
# 2. Momentum SGD
# ============================================================

class MomentumManual(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            mu = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p)
                v = state['velocity']
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                v.mul_(mu).add_(grad)
                p.add_(v, alpha=-group['lr'])
        return loss


# ============================================================
# 3. Adam (Baseline – correct)
# ============================================================

class AdamBaseline(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self.step_count += 1

        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g.add(p, alpha=group['weight_decay'])

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                m_hat = m / (1 - b1 ** self.step_count)
                v_hat = v / (1 - b2 ** self.step_count)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(group['eps']),
                           value=-group['lr'])
        return loss


# ============================================================
# 4. SR-Adam 
# ============================================================

class SRAdamAdaptiveLocal(Optimizer):
    """
    SR-Adam (Adaptive, Local, Whitened)

    - Stein shrinkage applied ONLY if group['stein'] == True
    - Shrinkage computed in Adam-whitened space
    - Local (per param_group) statistics
    - Warm-up + clipping for stability
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        warmup_steps=20,
        shrink_clip=(0.1, 1.0),
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            shrink_clip=shrink_clip,
            stein=True,   # default
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            stein_on = group.get('stein', True)
            warmup = group['warmup_steps']
            clip_lo, clip_hi = group['shrink_clip']

            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1
            step = group['step']

            grads, m_list, v_list = [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                grads.append(p.grad.view(-1))
                m_list.append(state['exp_avg'].view(-1))
                v_list.append(state['exp_avg_sq'].view(-1))

            if not grads:
                continue

            g = torch.cat(grads)
            m = torch.cat(m_list)
            v = torch.cat(v_list)

            # -------- Adam-whitened Stein shrinkage --------
            if (not stein_on) or (step <= warmup):
                shrink = 1.0
            else:
                # whitened quantities
                denom = (v.sqrt() + eps)
                g_w = g / denom
                m_w = m / denom

                sigma2 = (g_w - m_w).pow(2).mean().item()
                dist_sq = (g_w - m_w).pow(2).sum().item()
                p_dim = g_w.numel()

                raw = 1.0 - ((p_dim - 2) * sigma2) / (dist_sq + 1e-12)
                shrink = max(clip_lo, min(clip_hi, raw))

            # -------- Apply Adam update --------
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                m_t, v_t = state['exp_avg'], state['exp_avg_sq']

                g_hat = m_t + shrink * (grad - m_t)

                if group['weight_decay'] != 0:
                    g_hat = g_hat.add(p, alpha=group['weight_decay'])

                m_t.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)

                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bc2) / bc1

                denom = v_t.sqrt().add_(eps)
                p.addcdiv_(m_t, denom, value=-step_size)

        return loss
