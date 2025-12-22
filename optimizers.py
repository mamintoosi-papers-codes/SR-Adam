"""
Module: optimizers.py
Description: Optimizer implementations (SGD, Momentum, Adam, and SR-Adam variants)
"""

import torch
import torch.nn as nn
import math
from torch.optim import Optimizer


# ============================================================
# 1. SGD (Baseline)
# ============================================================

class SGDManual(Optimizer):
    """
    Vanilla Stochastic Gradient Descent
    """
    def __init__(self, params, lr=0.01, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                p.add_(grad, alpha=-lr)

        return loss


# ============================================================
# 2. SGD with Momentum (Baseline)
# ============================================================

class MomentumManual(Optimizer):
    """
    SGD with classical heavy-ball momentum
    """
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['velocity'] = torch.zeros_like(p)

                v = state['velocity']

                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                v.mul_(mu).add_(grad)
                p.add_(v, alpha=-lr)

        return loss


# ============================================================
# 3. Adam (Baseline – manual)
# ============================================================

class AdamBaseline(Optimizer):
    """
    Manual Adam implementation (baseline for fair comparison)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                m, v = state['exp_avg'], state['exp_avg_sq']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** self.global_step
                bc2 = 1 - beta2 ** self.global_step

                step_size = lr / bc1
                denom = (v.sqrt() / math.sqrt(bc2)).add_(group['eps'])

                p.addcdiv_(m, denom, value=-step_size)

        return loss


# ============================================================
# 4. SR-Adam (Fixed Sigma, Global)
# ============================================================

class SRAdamFixedGlobal(Optimizer):
    """
    Stein-Rule Adam with fixed sigma^2, global shrinkage
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, stein_sigma=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        stein_sigma=stein_sigma)
        super().__init__(params, defaults)
        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        grads, moms = [], []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                grads.append(p.grad.view(-1))
                moms.append(state['exp_avg'].view(-1))

        g = torch.cat(grads)
        m = torch.cat(moms)

        if self.global_step > 1:
            diff = g - m
            p_dim = g.numel()
            sigma2 = self.param_groups[0]['stein_sigma']
            shrink = 1 - (p_dim - 2) * sigma2 / (diff.pow(2).sum() + 1e-12)
            shrink = torch.clamp(shrink, 0.0, 1.0)
        else:
            shrink = 1.0

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                m_t, v_t = state['exp_avg'], state['exp_avg_sq']

                g_hat = m_t + shrink * (grad - m_t)

                m_t.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)

                bc1 = 1 - beta1 ** self.global_step
                bc2 = 1 - beta2 ** self.global_step
                step_size = lr / bc1
                denom = (v_t.sqrt() / math.sqrt(bc2)).add_(group['eps'])

                p.addcdiv_(m_t, denom, value=-step_size)

        return loss


# ============================================================
# 5. SR-Adam (Adaptive Sigma, Global)
# ============================================================

class SRAdamAdaptiveGlobal(Optimizer):
    """
    Stein-Rule Adam (Adaptive, Global, Stable)

    - Global James–Stein shrinkage
    - Noise variance estimated from Adam second moments
    - Warm-up + shrinkage clipping for stability
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        warmup_steps=20,
        shrink_clip=(0.1, 1.0)
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            shrink_clip=shrink_clip,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # -------- collect global buffers --------
        all_grad, all_m, all_v = [], [], []
        step = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                all_grad.append(p.grad.view(-1))
                all_m.append(state['exp_avg'].view(-1))
                all_v.append(state['exp_avg_sq'].view(-1))

                if step is None:
                    step = state['step']

        if len(all_grad) == 0:
            return loss

        g = torch.cat(all_grad)
        m = torch.cat(all_m)
        v = torch.cat(all_v)

        # -------- increment step --------
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['step'] += 1

        step += 1
        beta1, beta2 = self.param_groups[0]['betas']
        warmup = self.param_groups[0]['warmup_steps']
        clip_lo, clip_hi = self.param_groups[0]['shrink_clip']

        # -------- Stein shrinkage (global) --------
        if step <= warmup:
            shrink = 1.0
        else:
            # noise variance from Adam moments
            sigma2 = (v - m.pow(2)).clamp(min=0).mean().item()

            diff = g - m
            dist_sq = diff.pow(2).sum().item()
            p_dim = g.numel()

            raw = 1.0 - ((p_dim - 2) * sigma2) / (dist_sq + 1e-12)
            shrink = max(clip_lo, min(clip_hi, raw))

        # -------- apply Adam update --------
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                grad_hat = exp_avg + shrink * (grad - exp_avg)

                if group['weight_decay'] != 0:
                    grad_hat = grad_hat.add(p, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad_hat, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_hat, grad_hat, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ============================================================
# 6. SR-Adam (Adaptive Sigma, Local / Group-wise)
# ============================================================

class SRAdamAdaptiveLocal(Optimizer):
    """
    Stein-Rule Adam with adaptive sigma^2, local (per param_group) shrinkage
    
    FIXES APPLIED:
    - Uses Adam moments (v_t - m_t^2) to estimate sigma^2 correctly
    - Applies shrinkage clipping to prevent instability
    - Per-parameter-group step counting
    - Warm-up for first few steps to avoid early numerical issues
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0, warmup_steps=20, shrink_clip=(0.1, 1.0)):
        defaults = dict(lr=lr, betas=betas,
                        eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, shrink_clip=shrink_clip)
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
            warmup = group['warmup_steps']
            clip_lo, clip_hi = group['shrink_clip']

            # Initialize step counter for this group if needed
            if 'group_step' not in group:
                group['group_step'] = 0
            
            group['group_step'] += 1
            step = group['group_step']

            grads, moms, vars_list = [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                grads.append(p.grad.view(-1))
                moms.append(state['exp_avg'].view(-1))
                vars_list.append(state['exp_avg_sq'].view(-1))

            if not grads:
                continue

            g = torch.cat(grads)
            m = torch.cat(moms)
            v = torch.cat(vars_list)

            # -------- Compute adaptive shrinkage factor --------
            if step <= warmup:
                shrink = 1.0
            else:
                # Estimate noise variance from Adam moments: sigma^2 = E[g^2] - E[g]^2
                sigma2 = (v - m.pow(2)).clamp(min=0).mean().item()
                
                diff = g - m
                dist_sq = diff.pow(2).sum().item()
                p_dim = g.numel()
                
                # James-Stein shrinkage factor
                raw = 1.0 - ((p_dim - 2) * sigma2) / (dist_sq + 1e-12)
                
                # Clip to ensure stability
                shrink = max(clip_lo, min(clip_hi, raw))

            # -------- Apply updates to each parameter --------
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                m_t, v_t = state['exp_avg'], state['exp_avg_sq']

                # Stein-rule corrected gradient
                g_hat = m_t + shrink * (grad - m_t)

                if group['weight_decay'] != 0:
                    g_hat = g_hat.add(p, alpha=group['weight_decay'])

                # Update moments
                m_t.mul_(beta1).add_(g_hat, alpha=1 - beta1)
                v_t.mul_(beta2).addcmul_(g_hat, g_hat, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bc2) / bc1
                denom = v_t.sqrt().add_(group['eps'])

                p.addcdiv_(m_t, denom, value=-step_size)

        return loss
