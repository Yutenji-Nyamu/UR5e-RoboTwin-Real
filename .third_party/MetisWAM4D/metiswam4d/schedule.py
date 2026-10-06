"""Asynchronous noise schedule shared by training and inference.

One denoising progress ``u`` in [0, 1] drives every modality.  Modality ``m``
finishes at progress ``r_m`` and its (unshifted) noise level is

    sigma_m(u) = max(1 - u / r_m, 0),        r_action <= r_track <= r_video = 1.

Video therefore always carries at least as much noise as Track, and Track at
least as much as Action; the fastest modality reaches sigma = 0 first and then
acts as a clean condition for the others.  Modalities with the *same* completion
form one clock group and share a single noise level in every training mode
(e.g. ``r_action = r_track = 0.5``: Track and Action are denoised together and
the second half of the trajectory is video generation conditioned on the clean
action and the clean Track).  The same object produces the
training noise assignments (a mixture of the inference trajectory, ordered
independent draws and explicit clean edge cases) and the inference trajectory.

Noise levels are converted to the experts' flow-shifted sigma with

    shifted(sigma) = s * sigma / (1 + (s - 1) * sigma)

and experts receive ``timestep = 1000 * shifted(sigma)`` (Wan convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from metiswam4d import MODALITIES

TrainingMode = str
TRAINING_MODES: tuple[TrainingMode, ...] = (
    "trajectory",
    "ordered_independent",
    "independent",
    "clean_fastest",
    "clean_all_but_slowest",
)


def flow_shift(sigma: Tensor, shift: float) -> Tensor:
    """Shifted flow-matching noise level; identity for ``shift == 1``."""
    if shift <= 0:
        raise ValueError("flow shift must be positive")
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


@dataclass(frozen=True)
class AsyncSchedule:
    """Completion positions and flow shifts for the present modalities."""

    completion: dict[str, float] = field(
        default_factory=lambda: {"video": 1.0, "track": 0.75, "action": 0.25})
    shift: dict[str, float] = field(
        default_factory=lambda: {"video": 5.0, "track": 5.0, "action": 5.0})

    def __post_init__(self) -> None:
        for name in self.completion:
            if name not in MODALITIES:
                raise ValueError(f"unknown modality {name!r}")
            if not 0.0 < self.completion[name] <= 1.0:
                raise ValueError(f"completion of {name} must lie in (0, 1]")
            if name not in self.shift:
                raise ValueError(f"missing flow shift for {name}")
        if max(self.completion.values()) != 1.0:
            raise ValueError("the slowest modality must finish at progress 1.0")

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.completion)

    def present(self, present: tuple[str, ...] | None = None) -> "AsyncSchedule":
        """Restrict the schedule to a subset of experts (e.g. video + track)."""
        if present is None:
            return self
        missing = [name for name in present if name not in self.completion]
        if missing:
            raise ValueError(f"schedule lacks modalities {missing}")
        completion = {name: self.completion[name] for name in present}
        slowest = max(completion.values())
        if slowest != 1.0:
            # Re-anchor the slowest present modality at progress 1.
            completion = {name: value / slowest for name, value in completion.items()}
        return AsyncSchedule(completion, {name: self.shift[name] for name in present})

    def slow_to_fast(self) -> tuple[str, ...]:
        """Modalities ordered from the slowest (largest r) to the fastest."""
        return tuple(sorted(self.completion, key=lambda name: -self.completion[name]))

    def clock_groups(self) -> tuple[tuple[str, ...], ...]:
        """Modalities sharing a completion position, slowest group first; each group shares one noise level."""
        groups: list[list[str]] = []
        last: float | None = None
        for name in self.slow_to_fast():
            r = self.completion[name]
            if last is not None and abs(r - last) < 1e-9:
                groups[-1].append(name)
            else:
                groups.append([name])
                last = r
        return tuple(tuple(g) for g in groups)

    def sigma_at(self, u: Tensor, modality: str) -> Tensor:
        """Unshifted noise level of ``modality`` at progress ``u``."""
        r = self.completion[modality]
        return (1.0 - u / r).clamp(min=0.0, max=1.0)

    def shifted(self, sigma: Tensor, modality: str) -> Tensor:
        return flow_shift(sigma, self.shift[modality])

    def timestep(self, shifted_sigma: Tensor) -> Tensor:
        return shifted_sigma * 1000.0

    def inference_trajectory(self, rounds: int) -> dict[str, Tensor]:
        """Shifted sigma per modality on ``rounds + 1`` progress knots.

        Knot ``i`` corresponds to ``u = i / rounds``; entry ``0`` is pure noise
        and the last entry is zero for every modality.
        """
        if rounds < 1:
            raise ValueError("rounds must be positive")
        u = torch.linspace(0.0, 1.0, rounds + 1, dtype=torch.float64)
        return {
            name: self.shifted(self.sigma_at(u, name), name).float()
            for name in self.completion
        }

    def completion_round(self, modality: str, rounds: int) -> int:
        """First round index at which ``modality`` has reached sigma = 0."""
        import math
        return min(rounds, math.ceil(self.completion[modality] * rounds - 1e-9))


@dataclass
class NoiseAssignment:
    """Per-sample shifted noise levels for the present modalities."""

    sigma: dict[str, Tensor]
    mode: Tensor  # int64 [B], index into TRAINING_MODES

    def is_clean(self, modality: str) -> Tensor:
        return self.sigma[modality] <= 0.0

    def timestep(self, modality: str) -> Tensor:
        return self.sigma[modality] * 1000.0

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "NoiseAssignment":
        sigma = {
            name: value.to(device=device, dtype=dtype or value.dtype)
            for name, value in self.sigma.items()
        }
        return NoiseAssignment(sigma, self.mode.to(device))


@dataclass(frozen=True)
class NoiseMixture:
    """Training-time mixture over asynchronous noise configurations.

    ``trajectory`` samples ``u ~ U(0, 1)`` and reads every sigma from the
    inference trajectory, which naturally covers the "clean action, noisy
    world" and "clean action + clean track, noisy video" states.
    ``ordered_independent`` draws one i.i.d. base sigma per modality and sorts
    them so that the slower modality is noisier.  ``independent`` draws without
    ordering.  ``clean_fastest`` fixes the fastest present modality at zero and
    orders the rest; ``clean_all_but_slowest`` leaves only the slowest noisy.
    """

    weights: dict[TrainingMode, float] = field(default_factory=lambda: {
        "trajectory": 0.40,
        "ordered_independent": 0.40,
        "independent": 0.0,
        "clean_fastest": 0.15,
        "clean_all_but_slowest": 0.05,
    })
    min_noisy_sigma: float = 1e-3

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(TRAINING_MODES)
        if unknown:
            raise ValueError(f"unknown training modes {sorted(unknown)}")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("mixture weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("mixture weights must not all be zero")

    def _probabilities(self, schedule: AsyncSchedule) -> Tensor:
        weights = dict(self.weights)
        groups = len(schedule.clock_groups())
        if groups < 2:
            # A single clock has no asynchrony: everything is a plain draw.
            weights = {"independent": 1.0}
        elif groups == 2:
            # With two clocks the two clean edge cases coincide.
            weights["clean_fastest"] = weights.get("clean_fastest", 0.0) + \
                weights.pop("clean_all_but_slowest", 0.0)
        probs = torch.tensor(
            [weights.get(mode, 0.0) for mode in TRAINING_MODES], dtype=torch.float64)
        return probs / probs.sum()

    def sample(
        self,
        batch_size: int,
        schedule: AsyncSchedule,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> NoiseAssignment:
        order = schedule.slow_to_fast()
        groups = schedule.clock_groups()   # modalities of one group share a noise level
        n = len(groups)
        probs = self._probabilities(schedule)
        mode = torch.multinomial(probs, batch_size, replacement=True, generator=generator)
        u = torch.rand(batch_size, generator=generator, dtype=torch.float64)
        base = torch.rand(batch_size, n, generator=generator, dtype=torch.float64)
        ordered = base.sort(dim=1, descending=True).values  # slow -> fast

        sigma = {name: torch.zeros(batch_size, dtype=torch.float64) for name in order}

        def assign(b: int, index: int, value) -> None:
            for name in groups[index]:
                sigma[name][b] = value

        for b in range(batch_size):
            kind = TRAINING_MODES[int(mode[b])]
            if kind == "trajectory":
                for name in order:
                    sigma[name][b] = schedule.sigma_at(u[b], name)
            elif kind == "ordered_independent":
                for index in range(n):
                    assign(b, index, ordered[b, index])
            elif kind == "independent":
                for index in range(n):
                    assign(b, index, base[b, index])
            elif kind == "clean_fastest":
                for index in range(n - 1):
                    assign(b, index, ordered[b, index])
                assign(b, n - 1, 0.0)
            elif kind == "clean_all_but_slowest":
                assign(b, 0, base[b, 0])
                for index in range(1, n):
                    assign(b, index, 0.0)
            else:  # pragma: no cover
                raise RuntimeError(kind)

        shifted: dict[str, Tensor] = {}
        for name in order:
            value = sigma[name]
            noisy = value > 0
            value = torch.where(noisy, value.clamp(min=self.min_noisy_sigma), value)
            shifted[name] = schedule.shifted(value, name).float().to(device)
        return NoiseAssignment(shifted, mode.to(device))


def add_noise(clean: Tensor, sigma: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
    """Linear flow-matching path.  Returns ``(noisy, noise, velocity_target)``.

    ``sigma`` is ``[B]`` and broadcast over the trailing dimensions.
    """
    if noise is None:
        noise = torch.randn_like(clean)
    shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    s = sigma.to(dtype=clean.dtype, device=clean.device).view(shape)
    noisy = (1.0 - s) * clean + s * noise
    return noisy, noise, noise - clean


__all__ = [
    "AsyncSchedule",
    "NoiseAssignment",
    "NoiseMixture",
    "TRAINING_MODES",
    "add_noise",
    "flow_shift",
]
