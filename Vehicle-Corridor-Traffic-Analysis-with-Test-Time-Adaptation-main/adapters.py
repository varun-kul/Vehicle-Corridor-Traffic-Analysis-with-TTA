
# from __future__ import annotations
# import torch, torch.nn as nn

# def collect_bn_affine_params(model: nn.Module):
#     params = []
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#             if m.affine:
#                 params.append(m.weight)
#                 params.append(m.bias)
#     return params

# def set_bn_momentum(model: nn.Module, momentum: float = 0.1):
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#             m.momentum = momentum

# @torch.no_grad()
# def adabn_update(model: nn.Module, images_bchw, device: str = "cuda", max_batches: int = 64):
#     """Run images through the network in train() so BN running stats update, but without gradients."""
#     was_training = model.training
#     model.train()
#     set_bn_momentum(model, 0.01)
#     cnt = 0
#     for x in images_bchw:
#         if cnt >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
#         model(x)
#         cnt += 1
#     model.eval()
#     if was_training:
#         model.train()
#     return cnt

# # ---------- Additional TTA utilities: Tent-style and pseudo-label adaptation ----------
 
# def _entropy_loss_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Compute mean prediction entropy for detection head logits.
 
#     logits: (..., C) unnormalized class scores.
#     """
#     probs = torch.softmax(logits, dim=-1)
#     log_probs = torch.log(probs.clamp_min(eps))
#     ent = -(probs * log_probs).sum(dim=-1)    # (...,)
#     return ent.mean()
 
 
# @torch.no_grad()
# def snapshot_bn_state(model: nn.Module):
#     """
#     Take a shallow snapshot of BN running stats so we can roll back if needed.
#     Returns a dict mapping module id to (running_mean, running_var).
#     """
#     state = {}
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
#             state[id(m)] = (
#                 m.running_mean.detach().clone(),
#                 m.running_var.detach().clone(),
#             )
#     return state
 
 
# @torch.no_grad()
# def restore_bn_state(model: nn.Module, state) -> None:
#     """
#     Restore previously snapshotted BN running stats.
#     """
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and id(m) in state:
#             rm, rv = state[id(m)]
#             m.running_mean.copy_(rm)
#             m.running_var.copy_(rv)
 
 
# def tent_adapt(
#     model: nn.Module,
#     images_bchw,
#     device: torch.device | str = "cuda",
#     max_batches: int = 32,
#     steps_per_batch: int = 1,
#     lr: float = 1e-4,
#     weight_decay: float = 0.0,
#     rollback_if_worse: bool = True,
# ) -> int:
#     """
#     Entropy-minimization TTA (Tent-style) on top of a YOLOv8 detector.
 
#     images_bchw: iterable of tensors shaped [B, C, H, W] already on CPU.
#     Only BN affine parameters are updated; other weights stay frozen.
#     """
#     device = torch.device(device)
#     params = collect_bn_affine_params(model)
#     if not params:
#         return 0
 
#     was_training = model.training
#     model.train()
#     set_bn_momentum(model, 0.01)
 
#     opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
 
#     # Optional rollback safeguard: keep a copy of BN running stats
#     bn_snapshot = snapshot_bn_state(model) if rollback_if_worse else None
#     baseline_entropy = None
 
#     n_batches = 0
#     last_x = None
 
#     for i, x in enumerate(images_bchw):
#         if i >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
#         last_x = x
 
#         for _ in range(steps_per_batch):
#             opt.zero_grad(set_to_none=True)
#             out = model(x)
 
#             # Ultralytics YOLO forward typically returns (preds, feats, ...); take first tensor
#             if isinstance(out, (list, tuple)):
#                 preds = out[0]
#             else:
#                 preds = out
 
#             # preds: [B, N, no]; last classes are logits – skip 4 box coords + 1 obj score
#             if preds.ndim < 3 or preds.shape[-1] <= 5:
#                 cls_logits = preds.reshape(-1, preds.shape[-1])
#             else:
#                 cls_logits = preds[..., 5:].reshape(-1, preds.shape[-1] - 5)
 
#             loss = _entropy_loss_from_logits(cls_logits)
#             loss.backward()
#             opt.step()
 
#         # track entropy using the adapted model as a simple health proxy
#         with torch.no_grad():
#             out_eval = model(x)
#             if isinstance(out_eval, (list, tuple)):
#                 preds_eval = out_eval[0]
#             else:
#                 preds_eval = out_eval
#             if preds_eval.ndim < 3 or preds_eval.shape[-1] <= 5:
#                 cls_logits_eval = preds_eval.reshape(-1, preds_eval.shape[-1])
#             else:
#                 cls_logits_eval = preds_eval[..., 5:].reshape(
#                     -1, preds_eval.shape[-1] - 5
#                 )
#             ent_now = _entropy_loss_from_logits(cls_logits_eval)
 
#         if baseline_entropy is None:
#             baseline_entropy = ent_now.item()
#         n_batches += 1
 
#     # Simple rollback: if final entropy is higher than baseline, restore BN stats
#     if rollback_if_worse and bn_snapshot is not None and baseline_entropy is not None and last_x is not None:
#         with torch.no_grad():
#             out_final = model(last_x)
#             if isinstance(out_final, (list, tuple)):
#                 preds_final = out_final[0]
#             else:
#                 preds_final = out_final
#             if preds_final.ndim < 3 or preds_final.shape[-1] <= 5:
#                 cls_logits_final = preds_final.reshape(-1, preds_final.shape[-1])
#             else:
#                 cls_logits_final = preds_final[..., 5:].reshape(
#                     -1, preds_final.shape[-1] - 5
#                 )
#             ent_final = _entropy_loss_from_logits(cls_logits_final).item()
 
#         if ent_final > baseline_entropy:
#             restore_bn_state(model, bn_snapshot)
 
#     model.eval()
#     if was_training:
#         model.train()
#     return n_batches
 
 
# def pseudo_label_adapt(
#     model: nn.Module,
#     images_bchw,
#     device: torch.device | str = "cuda",
#     max_batches: int = 32,
#     steps_per_batch: int = 1,
#     lr: float = 1e-4,
#     weight_decay: float = 0.0,
#     conf_thr: float = 0.7,
# ) -> int:
#     """
#     Simple pseudo-label based TTA.
 
#     For each batch, we:
#       * Run the raw detector head to get class logits.
#       * Take the argmax class as a pseudo-label where max prob >= conf_thr.
#       * Minimize cross-entropy between logits and pseudo-labels, only on confident positions.
 
#     This adapts only BN affine parameters.
#     """
#     device = torch.device(device)
#     params = collect_bn_affine_params(model)
#     if not params:
#         return 0
 
#     was_training = model.training
#     model.train()
#     set_bn_momentum(model, 0.01)
 
#     opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
#     ce = torch.nn.CrossEntropyLoss(reduction="mean")
 
#     n_batches = 0
#     for i, x in enumerate(images_bchw):
#         if i >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
 
#         for _ in range(steps_per_batch):
#             opt.zero_grad(set_to_none=True)
#             out = model(x)
#             if isinstance(out, (list, tuple)):
#                 preds = out[0]
#             else:
#                 preds = out
 
#             if preds.ndim < 3 or preds.shape[-1] <= 5:
#                 cls_logits = preds.reshape(-1, preds.shape[-1])
#             else:
#                 cls_logits = preds[..., 5:].reshape(-1, preds.shape[-1] - 5)
 
#             probs = torch.softmax(cls_logits, dim=-1)
#             conf, pseudo = probs.max(dim=-1)
#             mask = conf >= conf_thr
#             if mask.sum() == 0:
#                 continue
 
#             loss = ce(cls_logits[mask], pseudo[mask])
#             loss.backward()
#             opt.step()
 
#         n_batches += 1
 
#     model.eval()
#     if was_training:
#         model.train()
#     return n_batches

#--------------------------

# from __future__ import annotations
 
# from typing import Iterable, Dict, Tuple
 
# import torch
# import torch.nn as nn
 
 
# # ---------------------------------------------------------------------
# # Basic BN helpers
# # ---------------------------------------------------------------------
 
 
# def collect_bn_layers(model: nn.Module) -> Iterable[nn.Module]:
#     """
#     Yield each BatchNorm layer (incl. SyncBatchNorm) in the model.
#     """
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
#             yield m
 
 
# def collect_bn_affine_params(model: nn.Module) -> Iterable[nn.Parameter]:
#     """
#     Collect only the affine parameters (weight/bias) of all BN layers.
 
#     These are the usual parameters adapted in Tent-style TTA.
#     """
#     params = []
#     for m in collect_bn_layers(model):
#         if m.affine:
#             if m.weight is not None:
#                 params.append(m.weight)
#             if m.bias is not None:
#                 params.append(m.bias)
#     return params
 
 
# def set_bn_momentum(model: nn.Module, momentum: float) -> None:
#     """
#     Set BatchNorm momentum (for running stats) to the given value.
#     """
#     for m in collect_bn_layers(model):
#         m.momentum = momentum
 
 
# # ---------------------------------------------------------------------
# # AdaBN: BN-stats-only adaptation
# # ---------------------------------------------------------------------
 
 
# @torch.no_grad()
# def adabn_update(
#     model: nn.Module,
#     images_bchw: Iterable[torch.Tensor],
#     device: torch.device | str = "cuda",
#     max_batches: int | None = None,
# ) -> int:
#     """
#     AdaBN-style test-time adaptation: update BN running stats using target images.
 
#     images_bchw: iterable of tensors [B, C, H, W] already on CPU or GPU.
#     Only BN running_mean / running_var are updated; weights are frozen.
#     """
#     device = torch.device(device)
#     was_training = model.training
 
#     # put BN into training mode (updates running stats) but don't track gradients
#     model.train()
#     set_bn_momentum(model, 0.01)
 
#     n_batches = 0
#     for i, x in enumerate(images_bchw):
#         if max_batches is not None and i >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
#         _ = model(x)  # forward just to update BN stats
#         n_batches += 1
 
#     model.eval()
#     if was_training:
#         model.train()
#     return n_batches
 
 
# # ---------------------------------------------------------------------
# # Tent-style entropy minimisation
# # ---------------------------------------------------------------------
 
 
# def _entropy_loss_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#     """
#     Mean prediction entropy from logits.
#     """
#     probs = torch.softmax(logits, dim=-1)
#     log_probs = torch.log(probs.clamp_min(eps))
#     ent = -(probs * log_probs).sum(dim=-1)
#     return ent.mean()
 
 
# @torch.no_grad()
# def snapshot_bn_state(model: nn.Module) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Snapshot BN running stats so we can roll back if adaptation gets worse.
#     Returns dict: module_id -> (running_mean, running_var).
#     """
#     state = {}
#     for m in collect_bn_layers(model):
#         state[id(m)] = (
#             m.running_mean.detach().clone(),
#             m.running_var.detach().clone(),
#         )
#     return state
 
 
# @torch.no_grad()
# def restore_bn_state(
#     model: nn.Module, state: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
# ) -> None:
#     """
#     Restore previously snapshotted BN running stats.
#     """
#     for m in collect_bn_layers(model):
#         mid = id(m)
#         if mid in state:
#             rm, rv = state[mid]
#             m.running_mean.copy_(rm)
#             m.running_var.copy_(rv)
 
 
# def tent_adapt(
#     model: nn.Module,
#     images_bchw: Iterable[torch.Tensor],
#     device: torch.device | str = "cuda",
#     max_batches: int = 32,
#     steps_per_batch: int = 1,
#     lr: float = 1e-4,
#     weight_decay: float = 0.0,
#     rollback_if_worse: bool = True,
# ) -> int:
#     """
#     Tent-style test-time adaptation: minimise prediction entropy via BN affine params.
 
#     * Only BN weight/bias are updated.
#     * Other network weights stay frozen.
#     * Uses a very simple entropy objective on detection head class logits.
 
#     Assumes YOLOv8-like detection head: final preds shape [B, N, 5 + C].
#     """
#     device = torch.device(device)
#     params = list(collect_bn_affine_params(model))
#     if not params:
#         return 0
 
#     was_training = model.training
#     model.train()
#     set_bn_momentum(model, 0.01)
 
#     opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
 
#     # Optional rollback
#     bn_snapshot = snapshot_bn_state(model) if rollback_if_worse else None
#     baseline_entropy = None
#     last_batch = None
 
#     n_batches = 0
#     for i, x in enumerate(images_bchw):
#         if i >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
#         last_batch = x
 
#         for _ in range(steps_per_batch):
#             opt.zero_grad(set_to_none=True)
#             out = model(x)
 
#             # YOLOv8 internal model: usually returns preds (B, N, 5 + C)
#             if isinstance(out, (list, tuple)):
#                 preds = out[0]
#             else:
#                 preds = out
 
#             if preds.ndim < 3 or preds.shape[-1] <= 5:
#                 cls_logits = preds.reshape(preds.shape[0], -1, preds.shape[-1])
#             else:
#                 cls_logits = preds[..., 5:]
 
#             loss = _entropy_loss_from_logits(cls_logits.reshape(-1, cls_logits.shape[-1]))
#             loss.backward()
#             opt.step()
 
#         # Track entropy after adaptation on this batch
#         with torch.no_grad():
#             out_eval = model(x)
#             if isinstance(out_eval, (list, tuple)):
#                 preds_eval = out_eval[0]
#             else:
#                 preds_eval = out_eval
 
#             if preds_eval.ndim < 3 or preds_eval.shape[-1] <= 5:
#                 cls_logits_eval = preds_eval.reshape(
#                     preds_eval.shape[0], -1, preds_eval.shape[-1]
#                 )
#             else:
#                 cls_logits_eval = preds_eval[..., 5:]
#             ent_now = _entropy_loss_from_logits(
#                 cls_logits_eval.reshape(-1, cls_logits_eval.shape[-1])
#             )
 
#         if baseline_entropy is None:
#             baseline_entropy = ent_now.item()
 
#         n_batches += 1
 
#     # Simple rollback: if entropy got worse, restore BN stats
#     if (
#         rollback_if_worse
#         and bn_snapshot is not None
#         and baseline_entropy is not None
#         and last_batch is not None
#     ):
#         with torch.no_grad():
#             out_final = model(last_batch)
#             if isinstance(out_final, (list, tuple)):
#                 preds_final = out_final[0]
#             else:
#                 preds_final = out_final
 
#             if preds_final.ndim < 3 or preds_final.shape[-1] <= 5:
#                 cls_logits_final = preds_final.reshape(
#                     preds_final.shape[0], -1, preds_final.shape[-1]
#                 )
#             else:
#                 cls_logits_final = preds_final[..., 5:]
#             ent_final = _entropy_loss_from_logits(
#                 cls_logits_final.reshape(-1, cls_logits_final.shape[-1])
#             ).item()
 
#         if ent_final > baseline_entropy:
#             restore_bn_state(model, bn_snapshot)
 
#     model.eval()
#     if was_training:
#         model.train()
#     return n_batches
 
 
# # ---------------------------------------------------------------------
# # Pseudo-label TTA
# # ---------------------------------------------------------------------
 
 
# def pseudo_label_adapt(
#     model: nn.Module,
#     images_bchw: Iterable[torch.Tensor],
#     device: torch.device | str = "cuda",
#     max_batches: int = 32,
#     steps_per_batch: int = 1,
#     lr: float = 1e-4,
#     weight_decay: float = 0.0,
#     conf_thr: float = 0.7,
# ) -> int:
#     """
#     Simple pseudo-label-based TTA.
 
#     For each batch:
#       * Get class logits from detection head.
#       * For positions with max prob >= conf_thr, take argmax as pseudo-label.
#       * Minimise CE between logits and pseudo-labels.
 
#     Only BN affine parameters are updated.
#     """
#     device = torch.device(device)
#     params = list(collect_bn_affine_params(model))
#     if not params:
#         return 0
 
#     was_training = model.training
#     model.train()
#     set_bn_momentum(model, 0.01)
 
#     opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
#     ce = torch.nn.CrossEntropyLoss(reduction="mean")
 
#     n_batches = 0
#     for i, x in enumerate(images_bchw):
#         if i >= max_batches:
#             break
#         x = x.to(device, non_blocking=True)
 
#         for _ in range(steps_per_batch):
#             opt.zero_grad(set_to_none=True)
#             out = model(x)
#             if isinstance(out, (list, tuple)):
#                 preds = out[0]
#             else:
#                 preds = out
 
#             if preds.ndim < 3 or preds.shape[-1] <= 5:
#                 cls_logits = preds.reshape(preds.shape[0], -1, preds.shape[-1])
#             else:
#                 cls_logits = preds[..., 5:]
 
#             cls_logits_flat = cls_logits.reshape(-1, cls_logits.shape[-1])
#             probs = torch.softmax(cls_logits_flat, dim=-1)
#             conf, pseudo = probs.max(dim=-1)
#             mask = conf >= conf_thr
#             if mask.sum() == 0:
#                 continue
 
#             loss = ce(cls_logits_flat[mask], pseudo[mask])
#             loss.backward()
#             opt.step()
 
#         n_batches += 1
 
#     model.eval()
#     if was_training:
#         model.train()
#     return n_batches

#-------------speed----------------

from __future__ import annotations

from typing import Iterable, Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Basic BN helpers
# ---------------------------------------------------------------------


def collect_bn_layers(model: nn.Module) -> Iterable[nn.Module]:
    """
    Yield each BatchNorm layer (incl. SyncBatchNorm) in the model.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            yield m


def collect_bn_affine_params(model: nn.Module) -> Iterable[nn.Parameter]:
    """
    Collect only the affine parameters (weight/bias) of all BN layers.

    These are the usual parameters adapted in Tent-style TTA.
    """
    params = []
    for m in collect_bn_layers(model):
        if m.affine:
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def set_bn_momentum(model: nn.Module, momentum: float) -> None:
    """
    Set BatchNorm momentum (for running stats) to the given value.
    """
    for m in collect_bn_layers(model):
        m.momentum = momentum


# ---------------------------------------------------------------------
# AdaBN: BN-stats-only adaptation
# ---------------------------------------------------------------------


@torch.no_grad()
def adabn_update(
    model: nn.Module,
    images_bchw: Iterable[torch.Tensor],
    device: torch.device | str = "cuda",
    max_batches: int | None = None,
) -> int:
    """
    AdaBN-style test-time adaptation: update BN running stats using target images.

    images_bchw: iterable of tensors [B, C, H, W] already on CPU or GPU.
    Only BN running_mean / running_var are updated; weights are frozen.
    """
    device = torch.device(device)
    was_training = model.training

    model.train()
    set_bn_momentum(model, 0.01)

    n_batches = 0
    for i, x in enumerate(images_bchw):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)  # forward just to update BN stats
        n_batches += 1

    model.eval()
    if was_training:
        model.train()
    return n_batches


# ---------------------------------------------------------------------
# Tent-style entropy minimisation
# ---------------------------------------------------------------------


def _entropy_loss_from_logits(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean prediction entropy from logits.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(eps))
    ent = -(probs * log_probs).sum(dim=-1)
    return ent.mean()


@torch.no_grad()
def snapshot_bn_state(model: nn.Module) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Snapshot BN running stats so we can roll back if adaptation gets worse.
    Returns dict: module_id -> (running_mean, running_var).
    """
    state = {}
    for m in collect_bn_layers(model):
        state[id(m)] = (
            m.running_mean.detach().clone(),
            m.running_var.detach().clone(),
        )
    return state


@torch.no_grad()
def restore_bn_state(
    model: nn.Module, state: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
) -> None:
    """
    Restore previously snapshotted BN running stats.
    """
    for m in collect_bn_layers(model):
        mid = id(m)
        if mid in state:
            rm, rv = state[mid]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)


def tent_adapt(
    model: nn.Module,
    images_bchw: Iterable[torch.Tensor],
    device: torch.device | str = "cuda",
    max_batches: int = 32,
    steps_per_batch: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    rollback_if_worse: bool = True,
) -> int:
    """
    Tent-style test-time adaptation: minimise prediction entropy via BN affine params.

    * Only BN weight/bias are updated.
    * Other network weights stay frozen.
    * Uses a simple entropy objective on detection head class logits.

    Assumes YOLOv8-like detection head: final preds shape [B, N, 5 + C].
    """
    device = torch.device(device)
    params = list(collect_bn_affine_params(model))
    if not params:
        return 0

    was_training = model.training
    model.train()
    set_bn_momentum(model, 0.01)

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    bn_snapshot = snapshot_bn_state(model) if rollback_if_worse else None
    baseline_entropy = None
    last_batch = None

    n_batches = 0
    for i, x in enumerate(images_bchw):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        last_batch = x

        for _ in range(steps_per_batch):
            opt.zero_grad(set_to_none=True)
            out = model(x)

            if isinstance(out, (list, tuple)):
                preds = out[0]
            else:
                preds = out

            if preds.ndim < 3 or preds.shape[-1] <= 5:
                cls_logits = preds.reshape(preds.shape[0], -1, preds.shape[-1])
            else:
                cls_logits = preds[..., 5:]

            loss = _entropy_loss_from_logits(
                cls_logits.reshape(-1, cls_logits.shape[-1])
            )
            loss.backward()
            opt.step()

        with torch.no_grad():
            out_eval = model(x)
            if isinstance(out_eval, (list, tuple)):
                preds_eval = out_eval[0]
            else:
                preds_eval = out_eval

            if preds_eval.ndim < 3 or preds_eval.shape[-1] <= 5:
                cls_logits_eval = preds_eval.reshape(
                    preds_eval.shape[0], -1, preds_eval.shape[-1]
                )
            else:
                cls_logits_eval = preds_eval[..., 5:]
            ent_now = _entropy_loss_from_logits(
                cls_logits_eval.reshape(-1, cls_logits_eval.shape[-1])
            )

        if baseline_entropy is None:
            baseline_entropy = ent_now.item()

        n_batches += 1

    if (
        rollback_if_worse
        and bn_snapshot is not None
        and baseline_entropy is not None
        and last_batch is not None
    ):
        with torch.no_grad():
            out_final = model(last_batch)
            if isinstance(out_final, (list, tuple)):
                preds_final = out_final[0]
            else:
                preds_final = out_final

            if preds_final.ndim < 3 or preds_final.shape[-1] <= 5:
                cls_logits_final = preds_final.reshape(
                    preds_final.shape[0], -1, preds_final.shape[-1]
                )
            else:
                cls_logits_final = preds_final[..., 5:]
            ent_final = _entropy_loss_from_logits(
                cls_logits_final.reshape(-1, cls_logits_final.shape[-1])
            ).item()

        if ent_final > baseline_entropy:
            restore_bn_state(model, bn_snapshot)

    model.eval()
    if was_training:
        model.train()
    return n_batches


# ---------------------------------------------------------------------
# Pseudo-label TTA
# ---------------------------------------------------------------------


def pseudo_label_adapt(
    model: nn.Module,
    images_bchw: Iterable[torch.Tensor],
    device: torch.device | str = "cuda",
    max_batches: int = 32,
    steps_per_batch: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    conf_thr: float = 0.7,
) -> int:
    """
    Simple pseudo-label-based TTA.

    For each batch:
      * Get class logits from detection head.
      * For positions with max prob >= conf_thr, take argmax as pseudo-label.
      * Minimise CE between logits and pseudo-labels.

    Only BN affine parameters are updated.
    """
    device = torch.device(device)
    params = list(collect_bn_affine_params(model))
    if not params:
        return 0

    was_training = model.training
    model.train()
    set_bn_momentum(model, 0.01)

    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss(reduction="mean")

    n_batches = 0
    for i, x in enumerate(images_bchw):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)

        for _ in range(steps_per_batch):
            opt.zero_grad(set_to_none=True)
            out = model(x)
            if isinstance(out, (list, tuple)):
                preds = out[0]
            else:
                preds = out

            if preds.ndim < 3 or preds.shape[-1] <= 5:
                cls_logits = preds.reshape(preds.shape[0], -1, preds.shape[-1])
            else:
                cls_logits = preds[..., 5:]

            cls_logits_flat = cls_logits.reshape(-1, cls_logits.shape[-1])
            probs = torch.softmax(cls_logits_flat, dim=-1)
            conf, pseudo = probs.max(dim=-1)
            mask = conf >= conf_thr
            if mask.sum() == 0:
                continue

            loss = ce(cls_logits_flat[mask], pseudo[mask])
            loss.backward()
            opt.step()

        n_batches += 1

    model.eval()
    if was_training:
        model.train()
    return n_batches
