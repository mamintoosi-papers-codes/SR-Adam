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
# 4. SR-Adam (Fixed Sigma, Global, WHITENED)
# ============================================================

class SRAdamFixedGlobal(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, stein_sigma=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, stein_sigma=stein_sigma)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self.step_count += 1

        # ---- collect global whitened stats ----
        gw, mw = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']
                denom = v.sqrt().add_(group['eps'])
                gw.append((p.grad / denom).view(-1))
                mw.append((m / denom).view(-1))

        if self.step_count > 1:
            g = torch.cat(gw)
            m = torch.cat(mw)
            diff = g - m
            p_dim = g.numel()
            sigma2 = self.param_groups[0]['stein_sigma']
            shrink = 1 - (p_dim - 2) * sigma2 / (diff.pow(2).sum() + 1e-12)
            shrink = torch.clamp(shrink, 0.0, 1.0)
        else:
            shrink = 1.0

        # ---- Adam update ----
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m, v = state['m'], state['v']
                g = p.grad

                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = v.sqrt().add_(group['eps'])

                g_hat = m + shrink * (g - m)
                m.mul_(b1).add_(g_hat, alpha=1 - b1)

                m_hat = m / (1 - b1 ** self.step_count)
                v_hat = v / (1 - b2 ** self.step_count)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(group['eps']),
                           value=-group['lr'])
        return loss


# ============================================================
# 5. SR-Adam (Adaptive Sigma, Global, WHITENED)
# ============================================================

class SRAdamAdaptiveGlobal(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0,
                 warmup_steps=20, shrink_clip=(0.1, 1.0)):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps,
                        shrink_clip=shrink_clip)
        super().__init__(params, defaults)
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self.step_count += 1

        gw, mw, vw = [], [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m, v = state['m'], state['v']
                denom = v.sqrt().add_(group['eps'])
                gw.append((p.grad / denom).view(-1))
                mw.append((m / denom).view(-1))
                vw.append((v / denom.pow(2)).view(-1))

        if self.step_count <= self.param_groups[0]['warmup_steps']:
            shrink = 1.0
        else:
            g = torch.cat(gw)
            m = torch.cat(mw)
            v = torch.cat(vw)
            sigma2 = (v - m.pow(2)).clamp(min=0).mean().item()
            diff = g - m
            raw = 1 - (g.numel() - 2) * sigma2 / (diff.pow(2).sum() + 1e-12)
            lo, hi = self.param_groups[0]['shrink_clip']
            shrink = max(lo, min(hi, raw))

        # Adam update
        for group in self.param_groups:
            b1, b2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                m, v = state['m'], state['v']
                g = p.grad
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                g_hat = m + shrink * (g - m)
                m.mul_(b1).add_(g_hat, alpha=1 - b1)

                m_hat = m / (1 - b1 ** self.step_count)
                v_hat = v / (1 - b2 ** self.step_count)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(group['eps']),
                           value=-group['lr'])
        return loss

