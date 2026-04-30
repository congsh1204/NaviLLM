"""
DEBUG_NAN 调试开关与辅助函数。

启用方式（任一即可）：
  - 命令行：`--debug`
  - 环境变量：`DEBUG_NAN=1`

推荐排查顺序：
  1) 单 GPU 开启 DEBUG_NAN 跑；
  2) 在策略采样前检查 nav_logits / nav_mask / softmax 诊断用 nav_probs；
  3) 若全景平均的分母 den=0（pano_masks 行和为 0）或无合法候选（valid_actions=0），做小范围修补；
  4) nav_logits 已非有限 → 回看 navigation 之前的 panorama 表征是否异常；
  5) forward 正常但某项 loss 非有限 → 按组件打印（navigation CE / obj / lm 等）；
  6) loss 有限但反向之后梯度非有限 → 查梯度与半精度（本仓库多在模型侧直转 fp16，非 torch.amp scaler）；
  7) 单卡稳定后，再加到 4 卡；
  8) 4 卡稳定后，再加到 8 卡。
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

import torch


_TRUTHY = frozenset(("1", "true", "yes", "on", "y"))


def debug_nan_from_args(args: Any) -> bool:
    if getattr(args, "debug", False):
        return True
    v = os.environ.get("DEBUG_NAN", "").strip().lower()
    return v in _TRUTHY


def dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def dist_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def dbg_print(msg: str) -> None:
    print(msg, flush=True)


def dbg_nonfinite_tensors(name: str, tensors: Iterable[tuple[str, torch.Tensor]], rank_only: Optional[int] = None) -> None:
    rk = dist_rank()
    if rank_only is not None and rk != rank_only:
        return
    for label, x in tensors:
        if isinstance(x, torch.Tensor) and x.numel() > 0 and not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            dbg_print(f"[DEBUG_NAN][rank={rk}] {name}: `{label}` has {bad}/{x.numel()} non-finite elements")


def pano_denominator_for_avg(pano_masks: torch.Tensor, args: Any) -> torch.Tensor:
    """与原先 `sum(pano_masks,1,keepdim=True)` 一致；DEBUG_NAN 下若 den=0 则 clamp 为 1 并打印。"""
    den = torch.sum(pano_masks, 1, keepdim=True)
    if debug_nan_from_args(args):
        dead = den.squeeze(-1) == 0
        if dead.any():
            dbg_print(
                f"[DEBUG_NAN][rank={dist_rank()}] pano_masks row sum zero (den=0) rows={dead.nonzero(as_tuple=False).squeeze(-1).tolist()} -> clamp(min=1)"
            )
            den = den.clamp(min=1)
    return den


def nav_candidate_mask(nav_inputs: dict) -> torch.Tensor:
    return nav_inputs["gmap_masks"] & (~nav_inputs["gmap_visited_masks"])


def maybe_patch_zero_valid_navigation(
    args: Any,
    nav_logits: torch.Tensor,
    ended: Iterable,
    nav_mask: torch.Tensor,
    batch_size: int,
    tag: str = "navigation",
) -> bool:
    """
    仍在进行的轨迹若 valid_actions=0，将无法合理采样；强制仅 stop（图索引 0）可用。
    返回是否在任一行上做过修补。
    """
    if not debug_nan_from_args(args):
        return False
    patched = False
    rk = dist_rank()
    mini = torch.finfo(nav_logits.dtype).min / 16
    for i in range(batch_size):
        if bool(ended[i]):
            continue
        nz = int(nav_mask[i].sum().item())
        if nz == 0:
            dbg_print(
                f"[DEBUG_NAN][rank={rk}] {tag}: batch_ix={i} valid_actions=0 (ended=False) -> force stop logits at index 0"
            )
            nav_logits[i].fill_(mini)
            nav_logits[i, 0] = 0.0
            patched = True
    return patched


def print_nav_sampling_diag(
    args: Any,
    nav_logits: torch.Tensor,
    nav_logits_sample: torch.Tensor,
    nav_mask: torch.Tensor,
    step_t: Optional[int],
) -> None:
    """在 sample/teacher 选择动作之前打印诊断（含 softmax(nav_logits_sample) 作为 nav_probs 仅用于观察）。"""
    if not debug_nan_from_args(args):
        return
    rk = dist_rank()
    with torch.no_grad():
        probs = torch.softmax(nav_logits_sample, dim=-1)
        row_sum = probs.sum(dim=-1)
        valid = nav_mask.sum(dim=-1)
        dbg_print(
            f"[DEBUG_NAN][rank={rk}] t={step_t} nav_mask valid_actions per row: {valid.tolist()}"
        )
        dbg_print(
            f"[DEBUG_NAN][rank={rk}] t={step_t} softmax(row_sum) should be ~1 on finite rows: {row_sum.tolist()}"
        )
        if not torch.isfinite(nav_logits).all():
            dbg_print(f"[DEBUG_NAN][rank={rk}] t={step_t} nav_logits non-finite (inspect panorama / nav forward above)")
        if not torch.isfinite(probs).all():
            dbg_print(f"[DEBUG_NAN][rank={rk}] t={step_t} nav_probs (diag softmax) non-finite")
        dbg_nonfinite_tensors(
            f"nav t={step_t}",
            [("nav_logits", nav_logits), ("nav_logits_sample", nav_logits_sample), ("diag_nav_probs", probs)],
        )


def report_loss_nonfinite(loss_name: str, loss_tensor: torch.Tensor) -> None:
    if torch.isfinite(loss_tensor).all():
        return
    rk = dist_rank()
    dbg_print(f"[DEBUG_NAN][rank={rk}] Loss `{loss_name}` is non-finite: value={loss_tensor.detach().float().cpu().item()}")


def check_gradients(model: torch.nn.Module, max_bad_print: int = 30) -> list[str]:
    """返回梯度含 NaN/Inf 的参数名前缀列表（为空表示未发现）。"""
    rk = dist_rank()
    bad: list[str] = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(n)
            if len(bad) <= max_bad_print:
                dbg_print(f"[DEBUG_NAN][rank={rk}] Bad grad (non-finite): {n}")
    if bad and len(bad) > max_bad_print:
        dbg_print(f"[DEBUG_NAN][rank={rk}] ... plus {len(bad) - max_bad_print} more parameters with bad grads")
    return bad


def warn_if_multi_gpu(logger, args: Any) -> None:
    if not debug_nan_from_args(args):
        return
    if dist_world_size() > 1:
        logger.warning(
            "DEBUG_NAN: 建议使用单 GPU 更易读日志；当前 world_size=%d。确认单卡稳定后再扩到 4/8 卡。",
            dist_world_size(),
        )
