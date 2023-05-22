"""Functional PyTorch definitions."""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from ..data import DataKeys
from ..distributions import Distribution

FINFO = torch.finfo()


def binary_mask_to_float_mask(mask: torch.Tensor, /) -> torch.Tensor:
    """Convert ``0`` and ``1`` elements in a binary mask to ``-inf`` and ``0``,
    respectively.

    Args:
        mask: Binary mask tensor.

    Returns:
        Float mask tensor where ``0`` indicates an UNPADDED or VALID value.

    """
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def float_mask_to_binary_mask(mask: torch.Tensor, /) -> torch.Tensor:
    """Convert ``0`` and ``-inf`` elements into a boolean mask of ``True`` and
    ``False``, respectively.

    Args:
        mask: Float mask tensor.

    Returns:
        Boolean mask tensor where ``True`` indicates an UNPADDED or VALID value.

    """
    return (
        mask.float()
        .masked_fill(mask == float("-inf"), False)
        .masked_fill(mask == 0, True)
        .bool()
    )


def generalized_advantage_estimate(
    batch: TensorDict,
    /,
    *,
    gae_lambda: float = 0.95,
    gamma: float = 0.95,
    inplace: bool = False,
    normalize: bool = True,
    return_returns: bool = True,
) -> TensorDict:
    """Compute a Generalized Advantage Estimate (GAE) and, optionally,
    returns using value function estimates and rewards.

    GAE is most commonly used with PPO for computing a policy loss that
    incentivizes "good" actions.

    Args:
        batch: Tensordict of batch size ``[B, T + 1, ...]`` that contains the
            following keys:

            - "rewards": Environment transition rewards.
            - "values": Policy value function estimates.

        gae_lambda: Generalized Advantage Estimation (GAE) hyperparameter for
            controlling the variance and bias tradeoff when estimating the
            state value function from collected environment transitions. A
            higher value allows higher variance while a lower value allows
            higher bias estimation but lower variance.
        gamma: Discount reward factor often used in the Bellman operator for
            controlling the variance and bias tradeoff in collected experienced
            rewards. Note, this does not control the bias/variance of the
            state value estimation and only controls the weight future rewards
            have on the total discounted return.
        inplace: Whether to store advantage and, optionally, return estimates
            in the given tensordict or whether to allocate a separate tensordict
            for the returned values.
        normalize: Whether to normalize advantages using the mean and standard
            deviation of the advantage batch before storing in the returned
            tensordict.
        return_returns: Whether to compute and return Monte Carlo return
            estimates with GAE.

    Returns:
        A tensordict with at least advantages and, optionally, discounted
        returns.

    """
    if inplace:
        out = batch
    else:
        out = TensorDict({}, batch_size=batch.batch_size, device=batch.device)
    if DataKeys.ADVANTAGES not in out.keys():
        out[DataKeys.ADVANTAGES] = torch.zeros_like(batch[DataKeys.REWARDS])
    prev_advantage = 0.0
    for t in reversed(range(batch.size(1) - 1)):
        delta = batch[DataKeys.REWARDS][:, t, ...] + (
            gamma * batch[DataKeys.VALUES][:, t + 1, ...]
            - batch[DataKeys.VALUES][:, t, ...]
        )
        out[DataKeys.ADVANTAGES][:, t, ...] = prev_advantage = delta + (
            gamma * gae_lambda * prev_advantage
        )
    if return_returns:
        out[DataKeys.RETURNS] = out[DataKeys.ADVANTAGES] + batch[DataKeys.VALUES]
    if normalize:
        std, mean = torch.std_mean(out[DataKeys.ADVANTAGES][:, :-1, ...])
        out[DataKeys.ADVANTAGES][:, :-1, ...] = (
            out[DataKeys.ADVANTAGES][:, :-1, ...] - mean
        ) / (std + 1e-8)
    return out


def mask_from_lengths(x: torch.Tensor, lengths: torch.Tensor, /) -> torch.Tensor:
    """Return sequence mask that indicates UNPADDED or VALID values
    according to tensor lengths.

    Args:
        x: Tensor with shape ``[B, T, ...]``.
        lengths: Tensor with shape ``[B]`` that indicates lengths of the
            ``T`` sequence for each B element in ``x``.

    Returns:
        Sequence mask of shape ``[B, T]``.

    """
    B, T = x.shape[:2]
    lengths = lengths.long().view(-1, 1).expand(B, T)
    range_tensor = torch.arange(T, device=lengths.device, dtype=lengths.dtype).expand(
        B, T
    )
    return range_tensor < lengths


def masked_avg(
    x: torch.Tensor,
    /,
    *,
    mask: None | torch.Tensor = None,
    dim: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    """Apply a masked average to ``x`` along ``dim``.

    Useful for pooling potentially padded features.

    Args:
        x: Tensor with shape ``[B, T, ...]`` to apply pooling to.
        mask: Mask with shape ``[B, T]`` indicating UNPADDED or VALID values.
        dim: Dimension to pool along.
        keepdim: Whether to keep the pooled dimension.

    Returns:
        Masked max of ``x`` along ``dim`` and the indices of those maximums.

    """
    if mask is not None:
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        masksum = mask.sum(dim=dim, keepdim=True)
        x = mask * x
        avg = x.sum(dim=dim, keepdim=True) / masksum
    else:
        avg = x.mean(dim=dim, keepdim=True)
    if not keepdim:
        avg = avg.squeeze(dim)
    return avg


def masked_categorical_sample(
    x: torch.Tensor, /, *, mask: None | torch.Tensor = None, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked categorical sampling of ``x``.

    Typically used for sampling from outputs of :meth:`masked_log_softmax`.

    Args:
        x: Logits with shape ``[B, T, ...]`` to sample from.
        mask: Mask with shape ``[B, T]`` indicating UNPADDED or VALID values.
        dim: Dimension to gather sampled values along.

    Returns:
        Sampled logits and the indices of those sampled logits.

    """
    if mask is not None:
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        x = x + torch.clamp(torch.log(mask), FINFO.min, FINFO.max)
    dist = torch.distributions.Categorical(logits=x)  # type: ignore[no-untyped-call]
    samples = dist.sample().unsqueeze(-1)  # type: ignore[no-untyped-call]
    return x.gather(dim, samples), samples


def masked_log_softmax(
    x: torch.Tensor, /, *, mask: None | torch.Tensor = None, dim: int = -1
) -> torch.Tensor:
    """Apply a masked log softmax to ``x`` along ``dim``.

    Typically used for getting logits from a model that predicts a sequence.
    The output of this function is typically passed to :meth:`masked_categorical_sample`.

    Args:
        x: Tensor with shape ``[B, T, ...]``.
        mask: Mask with shape ``[B, T]`` indicating UNPADDED or VALID values.
        dim: Dimension to apply log softmax along.

    Returns:
        Logits.

    """
    if mask is not None:
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        x = x + torch.clamp(torch.log(mask), FINFO.min, FINFO.max)
    return F.log_softmax(x, dim=dim)


def masked_max(
    x: torch.Tensor,
    /,
    *,
    mask: None | torch.Tensor = None,
    dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a masked max to ``x`` along ``dim``.

    Useful for pooling potentially padded features.

    Args:
        x: Tensor with shape ``[B, T, ...]`` to apply pooling to.
        mask: Mask with shape ``[B, T]`` indicating UNPADDED or VALID values.
        dim: Dimension to pool along.

    Returns:
        Masked max of ``x`` along ``dim`` and the indices of those maximums.

    """
    if mask is not None:
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask.bool(), FINFO.min)
    idx = x.argmax(dim=dim, keepdim=True)
    return x.gather(dim, idx), idx


def ppo_losses(
    buffer_batch: TensorDict,
    sample_batch: TensorDict,
    sample_distribution: Distribution,
    /,
    *,
    clip_param: float = 0.2,
    dual_clip_param: None | float = 5.0,
    entropy_coeff: float = 0.0,
    vf_clip_param: float = 1.0,
    vf_coeff: float = 1.0,
) -> TensorDict:
    """Proximal Policy Optimization loss.

    Includes a dual-clipped policy loss, value function estimate loss,
    and an optional entropy bonus loss. All losses are summed into a
    total loss and reduced with a mean operation.

    Args:
        buffer_batch: Tensordict of batch size ``[B, ...]`` full of the
            following keys:

            - "actions": Policy action samples during environment transitions.
            - "advantages": Advantages from :meth:`generalized_advantage_estimate`.
            - "logp": Log probabilities of taking ``"actions"``.
            - "returns": Monte carlo return estimates.

        sample_batch: Tensordict from sampling a policy of batch size ``[B, ...]``
            full of the following keys:

            - "values": Policy value function estimates.

        sample_distribution: A distribution instance created from the model
            that provided ``sample_batch`` used for computing the policy
            loss and entropy bonus loss.
        clip_param: PPO hyperparameter indicating the max distance the policy can
            update away from previously collected policy sample data with
            respect to likelihoods of taking actions conditioned on
            observations. This is the main innovation of PPO.
        dual_clip_param: PPO hyperparameter that clips like ``clip_param`` but when
            advantage estimations are negative. Helps prevent instability for
            continuous action spaces when policies are making large updates.
            Leave ``None`` for this clip to not apply. Otherwise, typical values
            are around ``5``.
        entropy_coeff: Entropy coefficient value. Weight of the entropy loss w.r.t.
            other components of total loss.
        vf_clip_param: PPO hyperparameter similar to ``clip_param`` but for
            the value function estimate. A measure of max distance the model's
            value function is allowed to update away from previous value
            function samples.
        vf_coeff: Value function loss component weight. Only needs to be tuned
            when the policy and value function share parameters.

    Returns:
        A tensordict containing each of the loss components.

    """
    p_ratio = torch.exp(
        sample_distribution.logp(buffer_batch[DataKeys.ACTIONS])
        - buffer_batch[DataKeys.LOGP]
    )
    vf_loss = torch.mean(
        torch.clamp(
            F.smooth_l1_loss(
                sample_batch[DataKeys.VALUES],
                buffer_batch[DataKeys.RETURNS],
                reduction="none",
            ),
            0.0,
            vf_clip_param,
        )
    )
    surr1 = buffer_batch[DataKeys.ADVANTAGES] * p_ratio
    surr2 = buffer_batch[DataKeys.ADVANTAGES] * torch.clamp(
        p_ratio, 1 - clip_param, 1 + clip_param
    )
    if dual_clip_param:
        clip1 = torch.min(surr1, surr2)
        clip2 = torch.max(
            clip1,
            dual_clip_param * buffer_batch[DataKeys.ADVANTAGES],
        )
        policy_loss = torch.where(
            buffer_batch[DataKeys.ADVANTAGES] < 0, clip2, clip1
        ).mean()
    else:
        policy_loss = torch.min(
            surr1,
            surr2,
        ).mean()
    total_loss = vf_coeff * vf_loss - policy_loss
    if entropy_coeff != 0:
        entropy_loss = sample_distribution.entropy().mean()
        total_loss -= entropy_coeff * entropy_loss
    else:
        entropy_loss = torch.tensor([0.0])
    return TensorDict(
        {
            "entropy": entropy_loss,
            "policy": policy_loss,
            "vf": vf_loss,
            "total": total_loss,
        },
        batch_size=[],
    )


def skip_connection(
    x: torch.Tensor,
    y: torch.Tensor,
    /,
    *,
    kind: None | str = "cat",
    dim: int = -1,
) -> torch.Tensor:
    """Perform a skip connection for ``x`` and ``y``.

    Args:
        x: Skip connection seed with shape ``[B, T, ...]``.
        y: Skip connection seed with same shape as ``x``.
        kind: Type of skip connection to use.
            Options include:

                - "residual" for a standard residual connection (summing outputs)
                - "cat" for concatenating outputs
                - ``None`` for no skip connection

        dim: Dimension to apply concatentation along. Only valid when
            ``kind`` is ``"cat"``

    Returns:
        A tensor with shape depending on ``kind``.

    """
    match kind:
        case "residual":
            return x + y
        case "cat":
            return torch.cat([x, y], dim=dim)
        case None:
            return y
    raise ValueError(f"No skip connection type for {kind}.")
