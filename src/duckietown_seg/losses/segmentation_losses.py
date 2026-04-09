from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def one_hot_encode(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()


def soft_dice_score(
    probabilities: torch.Tensor,
    targets_one_hot: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * targets_one_hot, dim=dims)
    cardinality = torch.sum(probabilities + targets_one_hot, dim=dims)
    return (2.0 * intersection + epsilon) / (cardinality + epsilon)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    targets_one_hot = one_hot_encode(targets, num_classes=num_classes)
    dice = soft_dice_score(probabilities, targets_one_hot, epsilon=epsilon)
    return 1.0 - dice.mean()


def dsc_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return dice_loss_from_logits(logits, targets, epsilon=epsilon)


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, targets, weight=class_weights)


def focal_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - target_probs).pow(gamma)
    loss = -focal_factor * target_log_probs
    if alpha is not None:
        alpha_factor = alpha.to(logits.device).gather(0, targets.view(-1)).view_as(targets)
        loss = loss * alpha_factor
    return loss.mean()


def estimate_class_weights(
    dataloader: torch.utils.data.DataLoader[Any],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, masks in dataloader:
        for class_id in range(num_classes):
            counts[class_id] += (masks == class_id).sum().item()
    counts = counts.clamp_min(1.0)
    frequencies = counts / counts.sum()
    weights = (1.0 / frequencies).to(torch.float32)
    weights = weights / weights.mean()
    return weights.to(device)


def estimate_focal_alpha(
    dataloader: torch.utils.data.DataLoader[Any],
    num_classes: int,
    device: torch.device,
    background_index: int = 0,
    background_dampen: float = 0.3,
) -> torch.Tensor:
    alpha = estimate_class_weights(dataloader, num_classes=num_classes, device=device)
    alpha = alpha / alpha.sum()
    alpha[background_index] = alpha[background_index] * background_dampen
    alpha = alpha / alpha.sum()
    return alpha


def build_loss(
    loss_name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    focal_alpha: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
    focal_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalized = loss_name.lower().replace("+", "_")
    metrics: dict[str, float] = {}

    if normalized == "cross_entropy":
        loss = cross_entropy_loss(logits, targets, class_weights)
    elif normalized == "focal_dice":
        dice_loss = dice_loss_from_logits(logits, targets)
        focal_loss = focal_loss_from_logits(logits, targets, alpha=focal_alpha, gamma=focal_gamma)
        loss = dice_weight * dice_loss + focal_weight * focal_loss
        metrics["dice_loss"] = float(dice_loss.detach().cpu())
        metrics["focal_loss"] = float(focal_loss.detach().cpu())
    elif normalized == "dice_ce":
        dice_loss = dice_loss_from_logits(logits, targets)
        ce_loss = cross_entropy_loss(logits, targets, class_weights)
        loss = dice_weight * dice_loss + ce_weight * ce_loss
        metrics["dice_loss"] = float(dice_loss.detach().cpu())
        metrics["cross_entropy_loss"] = float(ce_loss.detach().cpu())
    elif normalized == "dsc":
        loss = dsc_loss_from_logits(logits, targets)
    else:
        raise ValueError(f"Unsupported loss_name '{loss_name}'.")

    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics

