# Copyright (c) 2020-2026 The Center for Theoretical Biological Physics (CTBP)
# Rice University
# This file is part of OpenMiChroM and is released under the MIT License.

"""Deterministic one-dimensional loop-extrusion trajectories and bond updates.

Miles Gantcher contributed the first OpenMiChroM implementation of this
workflow in pull request 123.  The model is conceptually related to the loop
extrusion simulations described by Sanborn *et al.* (PNAS 2015,
doi:10.1073/pnas.1518552112).  This maintained implementation adds an explicit
zero-based convention, validation, bounded placement, collision invariants,
and seeded randomness.

``LoopBondUpdater`` activates a precomputed union of harmonic loop bonds.  It
updates only per-bond strengths in an existing OpenMM Context, which avoids
serializing coordinates or rebuilding a simulation at every extrusion step.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _probability_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain numeric values")
    result = np.asarray(array, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if ((result < 0.0) | (result > 1.0)).any():
        raise ValueError(f"{name} values must be between 0 and 1")
    return result.copy()


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class LoopExtrusionManager:
    """Generate a seeded two-sided loop-extrusion trajectory.

    Parameters
    ----------
    fprobs_fix, rprobs_fix
        Per-bead probabilities that the left or right extruder foot becomes
        fixed during one extrusion transition.  Arrays use zero-based bead
        indices and must have equal length.
    num_steps
        Number of extrusion transitions.  The returned trajectory includes the
        initial state and therefore has ``num_steps + 1`` frames.
    extruder_count
        Number of simultaneous two-foot extruders.
    off_rate
        Per-transition probability that an extruder unloads and is reloaded at
        another valid position.
    unfix_conversion_factor
        For a nonzero fixation probability ``p``, the per-transition unfixing
        probability is ``min(1, unfix_conversion_factor / p)``.
    seed
        Seed for NumPy's local ``Generator``.  The global NumPy random state is
        never used.
    boundary_buffer
        Number of beads excluded at each chromosome end.
    reload_policy
        Action when expanded non-reset loops leave no valid site for an
        unloaded motor. ``"collective"`` reloads all motors together;
        ``"error"`` stops instead of changing unaffected motors.

    Notes
    -----
    Extruder feet cannot occupy the same bead or pass through one another.  A
    collision or boundary encounter unloads the affected motor. If it cannot
    be reloaded around the remaining loops, ``reload_policy`` controls whether
    all motors are reloaded or the simulation stops. Nested loops are
    permitted; interleaving loops are not.
    """

    def __init__(
        self,
        fprobs_fix: ArrayLike,
        rprobs_fix: ArrayLike,
        num_steps: int,
        extruder_count: int,
        off_rate: float = 1 / 500,
        unfix_conversion_factor: float = 1 / 4_000,
        *,
        seed: int | None = None,
        boundary_buffer: int = 2,
        reload_policy: str = "collective",
    ) -> None:
        forward = _probability_vector(fprobs_fix, name="fprobs_fix")
        reverse = _probability_vector(rprobs_fix, name="rprobs_fix")
        if forward.shape != reverse.shape:
            raise ValueError("fprobs_fix and rprobs_fix must have the same length")

        self.num_steps = _integer(num_steps, name="num_steps", minimum=0)
        self.extruder_count = _integer(
            extruder_count, name="extruder_count", minimum=1
        )
        self.off_rate = _probability(off_rate, name="off_rate")
        self.unfix_conversion_factor = _nonnegative_real(
            unfix_conversion_factor, name="unfix_conversion_factor"
        )
        self.boundary_buffer = _integer(
            boundary_buffer, name="boundary_buffer", minimum=1
        )
        if not isinstance(reload_policy, str):
            raise TypeError("reload_policy must be a string")
        self.reload_policy = reload_policy.strip().lower()
        if self.reload_policy not in {"collective", "error"}:
            raise ValueError("reload_policy must be 'collective' or 'error'")
        self.collective_reload_count = 0
        self.chrom_length = int(forward.size)
        if self.chrom_length < 2 * self.boundary_buffer + 3:
            raise ValueError(
                "chromosome is too short for a two-foot extruder and the "
                f"requested boundary_buffer={self.boundary_buffer}"
            )

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise TypeError("seed must be an integer or None")
        self.seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.fprobs_fix = forward
        self.rprobs_fix = reverse
        self.fprobs_unfix = self._unfix_probabilities(forward)
        self.rprobs_unfix = self._unfix_probabilities(reverse)
        self._trajectory: NDArray[np.int64] | None = None

        # Fail promptly when even the initial non-overlapping placement cannot
        # fit the requested number of motors.
        possible_centers = np.arange(
            self.boundary_buffer + 1,
            self.chrom_length - self.boundary_buffer - 1,
        )
        maximum = (possible_centers.size + 2) // 3
        if self.extruder_count > maximum:
            raise ValueError(
                f"extruder_count={self.extruder_count} exceeds the placement "
                f"capacity {maximum} for chromosome length {self.chrom_length}"
            )

    def _unfix_probabilities(
        self, fixation: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.zeros_like(fixation)
        nonzero = fixation > 0.0
        result[nonzero] = np.minimum(
            1.0, self.unfix_conversion_factor / fixation[nonzero]
        )
        return result

    @staticmethod
    def _interleaves(
        pair: tuple[int, int], existing: list[tuple[int, int]]
    ) -> bool:
        left, right = pair
        for other_left, other_right in existing:
            one_new_inside = (other_left < left < other_right) != (
                other_left < right < other_right
            )
            one_old_inside = (left < other_left < right) != (
                left < other_right < right
            )
            if one_new_inside or one_old_inside:
                return True
        return False

    def _valid_centers(
        self, existing: list[tuple[int, int]]
    ) -> NDArray[np.int64]:
        occupied = np.asarray([anchor for pair in existing for anchor in pair], dtype=int)
        candidates: list[int] = []
        for center in range(
            self.boundary_buffer + 1,
            self.chrom_length - self.boundary_buffer - 1,
        ):
            if occupied.size and np.any(np.abs(occupied - center) <= 1):
                continue
            pair = (center - 1, center + 1)
            if not self._interleaves(pair, existing):
                candidates.append(center)
        return np.asarray(candidates, dtype=np.int64)

    def _place_bonds(
        self, existing: list[tuple[int, int]], count: int
    ) -> list[tuple[int, int]] | None:
        """Choose ``count`` mutually compatible reload sites, if possible."""
        if count == 0:
            return []
        candidates = self._valid_centers(existing)
        if candidates.size < count:
            return None

        # Dynamic programming gives the maximum number of centers that remain
        # after either taking or skipping each candidate.  Randomly choose
        # between feasible branches, but never make a greedy choice that can
        # prevent placement of a later motor.
        next_index = np.searchsorted(candidates, candidates + 3, side="left")
        maximum = np.zeros(candidates.size + 1, dtype=np.int64)
        for index in range(candidates.size - 1, -1, -1):
            maximum[index] = max(
                maximum[index + 1], 1 + maximum[next_index[index]]
            )
        if maximum[0] < count:
            return None

        selected: list[int] = []
        index = 0
        remaining = count
        while remaining:
            include_is_feasible = (
                1 + maximum[next_index[index]] >= remaining
            )
            skip_is_feasible = maximum[index + 1] >= remaining
            include = include_is_feasible and (
                not skip_is_feasible or bool(self._rng.integers(2))
            )
            if include:
                selected.append(int(candidates[index]))
                remaining -= 1
                index = int(next_index[index])
            else:
                index += 1

        result = [(center - 1, center + 1) for center in selected]
        self._rng.shuffle(result)
        return result

    def _initial_bonds(self) -> NDArray[np.int64]:
        # A sequential random placement can paint itself into a corner at the
        # exact advertised capacity (for example, by choosing the middle of
        # four available centers first).  Compress the required two-bead gaps,
        # sample in that smaller space, and expand again.  This produces a
        # complete non-overlapping placement whenever the capacity check in
        # ``__init__`` says one exists.
        possible_centers = np.arange(
            self.boundary_buffer + 1,
            self.chrom_length - self.boundary_buffer - 1,
            dtype=np.int64,
        )
        compressed_size = possible_centers.size - 2 * (self.extruder_count - 1)
        compressed = np.sort(
            self._rng.choice(
                compressed_size, size=self.extruder_count, replace=False
            )
        )
        centers = possible_centers[compressed + 2 * np.arange(self.extruder_count)]
        placed = np.column_stack((centers - 1, centers + 1)).astype(np.int64)
        self._rng.shuffle(placed)
        return placed

    @classmethod
    def _assert_frame_invariants(cls, frame: NDArray[np.int64]) -> None:
        anchors = frame.ravel()
        if np.unique(anchors).size != anchors.size:
            raise RuntimeError("internal error: extrusion frame contains colliding anchors")
        pairs = [tuple(map(int, pair)) for pair in frame]
        if any(left >= right for left, right in pairs):
            raise RuntimeError("internal error: extrusion frame has an invalid bond")
        for index, pair in enumerate(pairs):
            if cls._interleaves(pair, pairs[:index] + pairs[index + 1 :]):
                raise RuntimeError("internal error: extrusion frame contains crossing bonds")

    def _generate(self) -> NDArray[np.int64]:
        bonds = np.empty(
            (self.num_steps + 1, self.extruder_count, 2), dtype=np.int64
        )
        bonds[0] = self._initial_bonds()
        fixed = np.zeros((self.extruder_count, 2), dtype=bool)
        lower = self.boundary_buffer
        upper = self.chrom_length - self.boundary_buffer - 1

        for step in range(self.num_steps):
            current = bonds[step]
            proposed = current.copy()
            reset = np.zeros(self.extruder_count, dtype=bool)

            for index, (left, right) in enumerate(current):
                if self._rng.random() < self.fprobs_fix[left]:
                    fixed[index, 0] = True
                if self._rng.random() < self.rprobs_fix[right]:
                    fixed[index, 1] = True
                if fixed[index, 0] and self._rng.random() < self.fprobs_unfix[left]:
                    fixed[index, 0] = False
                if fixed[index, 1] and self._rng.random() < self.rprobs_unfix[right]:
                    fixed[index, 1] = False

                if not fixed[index, 0]:
                    proposed[index, 0] = left - 1
                if not fixed[index, 1]:
                    proposed[index, 1] = right + 1
                if proposed[index, 0] < lower or proposed[index, 1] > upper:
                    reset[index] = True
                if self._rng.random() < self.off_rate:
                    reset[index] = True

            owner_by_anchor = {
                int(anchor): owner
                for owner, pair in enumerate(current)
                for anchor in pair
            }
            for owner, pair in enumerate(proposed):
                if reset[owner]:
                    continue
                for foot, anchor in enumerate(pair):
                    if anchor == current[owner, foot]:
                        continue
                    other = owner_by_anchor.get(int(anchor))
                    if other is not None and other != owner:
                        reset[owner] = True
                        reset[other] = True

            proposed_owners: dict[int, list[int]] = {}
            for owner, pair in enumerate(proposed):
                if reset[owner]:
                    continue
                for anchor in pair:
                    proposed_owners.setdefault(int(anchor), []).append(owner)
            for owners in proposed_owners.values():
                if len(owners) > 1:
                    reset[owners] = True

            for first in range(self.extruder_count):
                if reset[first]:
                    continue
                for second in range(first + 1, self.extruder_count):
                    if reset[second]:
                        continue
                    if self._interleaves(
                        tuple(map(int, proposed[first])),
                        [tuple(map(int, proposed[second]))],
                    ):
                        reset[first] = True
                        reset[second] = True

            next_pairs: list[tuple[int, int] | None] = [
                None if reset[index] else tuple(map(int, proposed[index]))
                for index in range(self.extruder_count)
            ]
            reset_owners = np.flatnonzero(reset)
            existing = [pair for pair in next_pairs if pair is not None]
            replacements = self._place_bonds(existing, int(reset_owners.size))

            # At very high density, expanded non-reset loops can temporarily
            # leave no legal loading site.  Treat that state as a collective
            # collision and reload the full motor population in one bounded,
            # capacity-aware operation.
            if replacements is None and self.reload_policy == "collective":
                reset_owners = np.arange(self.extruder_count)
                next_pairs = [None] * self.extruder_count
                fixed[:] = False
                replacements = self._place_bonds([], self.extruder_count)
                self.collective_reload_count += 1
            if replacements is None:
                raise RuntimeError(
                    "no legal reload sites remain around the active loops; "
                    "use reload_policy='collective' to reload all motors"
                )
            for owner, replacement in zip(reset_owners, replacements):
                next_pairs[int(owner)] = replacement
                fixed[owner] = False

            frame = np.asarray(next_pairs, dtype=np.int64)
            self._assert_frame_invariants(frame)
            bonds[step + 1] = frame

        return bonds

    def simulate(self) -> NDArray[np.int64]:
        """Return the cached trajectory, generating it on the first call."""

        if self._trajectory is None:
            self._trajectory = self._generate()
        return self._trajectory.copy()

    def get_extrusion_bonds(self) -> NDArray[np.int64]:
        """Compatibility name for :meth:`simulate`."""

        return self.simulate()


class LoopBondUpdater:
    """Activate a loop trajectory in one existing OpenMM Context.

    The force contains the union of all particle pairs in ``trajectory``.  A
    per-bond ``active`` parameter is set to one for the requested frame and zero
    for every other pair, so OpenMM can update the Context without changing the
    force topology.
    """

    def __init__(
        self,
        trajectory: ArrayLike,
        *,
        k_loop: float = 10.0,
        r0_loop: float = 1.0,
    ) -> None:
        raw = np.asarray(trajectory)
        if raw.ndim == 2 and raw.shape[-1] == 2:
            raw = raw[np.newaxis, ...]
        if raw.ndim != 3 or raw.shape[-1] != 2:
            raise ValueError("trajectory must have shape (frames, extruders, 2)")
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError("trajectory must contain at least one frame and one bond")
        if not np.issubdtype(raw.dtype, np.integer):
            if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
                raise TypeError("trajectory must contain integer particle indices")
            if not np.equal(raw, np.floor(raw)).all():
                raise TypeError("trajectory must contain integer particle indices")
        normalized = np.asarray(raw, dtype=np.int64)
        if (normalized < 0).any():
            raise ValueError("trajectory particle indices must be non-negative")
        normalized = np.sort(normalized, axis=-1)
        if (normalized[..., 0] == normalized[..., 1]).any():
            raise ValueError("a loop bond cannot connect a particle to itself")
        for frame in normalized:
            pairs = [tuple(map(int, pair)) for pair in frame]
            if len(pairs) != len(set(pairs)):
                raise ValueError("a trajectory frame cannot contain duplicate loop bonds")

        self.trajectory = normalized
        self.k_loop = _nonnegative_real(k_loop, name="k_loop")
        self.r0_loop = _nonnegative_real(r0_loop, name="r0_loop")
        self.pairs = tuple(
            sorted({tuple(map(int, pair)) for frame in normalized for pair in frame})
        )
        self._pair_to_index = {pair: index for index, pair in enumerate(self.pairs)}
        self.force: Any | None = None
        self.current_step: int | None = None

    @property
    def particle_count(self) -> int:
        """Smallest particle count that can contain the trajectory."""

        return max(anchor for pair in self.pairs for anchor in pair) + 1

    def active_pairs(self, step: int) -> tuple[tuple[int, int], ...]:
        """Return the unique active particle pairs for one trajectory frame."""

        index = _integer(step, name="step", minimum=0)
        if index >= self.trajectory.shape[0]:
            raise IndexError(
                f"step {index} is outside a {self.trajectory.shape[0]}-frame trajectory"
            )
        return tuple(sorted({tuple(map(int, pair)) for pair in self.trajectory[index]}))

    def create_force(self, openmm_module: Any) -> Any:
        """Create the OpenMM ``CustomBondForce`` for this trajectory."""

        if self.force is not None:
            raise RuntimeError("the loop force has already been created")
        force = openmm_module.CustomBondForce(
            "active*0.5*k_loop*(r-r0_loop)^2"
        )
        force.setName("LoopExtrusion")
        force.addPerBondParameter("active")
        # Keep the physical constants per bond as well.  OpenMM requires
        # identically named global parameters to share one default value across
        # every force in a Context, which would make two independently tuned
        # loop trajectories conflict with each other.
        force.addPerBondParameter("k_loop")
        force.addPerBondParameter("r0_loop")
        initial = set(self.active_pairs(0))
        for particle1, particle2 in self.pairs:
            force.addBond(
                particle1,
                particle2,
                [
                    1.0 if (particle1, particle2) in initial else 0.0,
                    self.k_loop,
                    self.r0_loop,
                ],
            )
        self.force = force
        self.current_step = 0
        return force

    def set_step(self, step: int, context: Any) -> None:
        """Activate ``step`` in an already-created OpenMM ``Context``."""

        if self.force is None:
            raise RuntimeError("create_force() must be called before set_step()")
        active = set(self.active_pairs(step))
        for pair, index in self._pair_to_index.items():
            self.force.setBondParameters(
                index,
                pair[0],
                pair[1],
                [1.0 if pair in active else 0.0, self.k_loop, self.r0_loop],
            )
        self.force.updateParametersInContext(context)
        self.current_step = int(step)


# The original PR used this public spelling.  Keep it as a true alias so code
# written against that branch can move to the maintained implementation.
Loop_Extrusion_Manager = LoopExtrusionManager


__all__ = ["LoopBondUpdater", "LoopExtrusionManager", "Loop_Extrusion_Manager"]
