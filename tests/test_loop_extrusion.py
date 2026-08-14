import numpy as np
import openmm
import pytest
from openmm import unit

import OpenMiChroM
from OpenMiChroM.Extrusion_Bonds import (
    LoopBondUpdater,
    LoopExtrusionManager,
    Loop_Extrusion_Manager,
)
from OpenMiChroM.ChromDynamics import MiChroM


def _manager(length=20, **kwargs):
    return LoopExtrusionManager(
        np.zeros(length),
        np.zeros(length),
        num_steps=kwargs.pop("num_steps", 4),
        extruder_count=kwargs.pop("extruder_count", 2),
        off_rate=kwargs.pop("off_rate", 0.0),
        unfix_conversion_factor=kwargs.pop("unfix_conversion_factor", 0.0),
        seed=kwargs.pop("seed", 123),
        **kwargs,
    )


def _manager_with_blocked_local_reload(reload_policy="collective"):
    forward = np.zeros(10)
    reverse = np.zeros(10)
    forward[3] = 1.0
    reverse[5] = 1.0
    manager = LoopExtrusionManager(
        forward,
        reverse,
        num_steps=1,
        extruder_count=2,
        off_rate=0.0,
        unfix_conversion_factor=0.0,
        seed=0,
        reload_policy=reload_policy,
    )
    initial = np.array([[2, 6], [3, 5]], dtype=np.int64)
    # This is a valid nested frame that directly exercises the policy state:
    # the inner loop fixes in place while the outer loop reaches both bounds,
    # leaving no legal site for a local reload.
    manager._initial_bonds = lambda: initial.copy()
    return manager, initial


def _assert_no_interleaving(pairs):
    for first, (left, right) in enumerate(pairs):
        for other_left, other_right in pairs[first + 1 :]:
            assert not (left < other_left < right < other_right)
            assert not (other_left < left < other_right < right)


def _bond_parameters(force):
    return [
        (
            int(force.getBondParameters(index)[0]),
            int(force.getBondParameters(index)[1]),
            float(force.getBondParameters(index)[2][0]),
        )
        for index in range(force.getNumBonds())
    ]


def test_seeded_trajectory_is_exact_and_cached_defensively():
    manager = _manager()
    expected = np.array(
        [
            [[2, 4], [12, 14]],
            [[2, 4], [11, 15]],
            [[2, 4], [10, 16]],
            [[10, 12], [9, 17]],
            [[2, 4], [5, 7]],
        ],
        dtype=np.int64,
    )

    first = manager.simulate()
    first[0, 0, 0] = 999
    second = manager.simulate()

    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(manager.get_extrusion_bonds(), expected)
    assert second.dtype == np.int64


def test_seed_is_local_and_does_not_consume_numpy_global_random_state():
    np.random.seed(2468)
    expected = np.random.random(4)
    np.random.seed(2468)

    _manager(seed=99).simulate()
    observed = np.random.random(4)

    np.testing.assert_array_equal(observed, expected)


def test_all_frames_obey_zero_based_bound_collision_and_noncrossing_invariants():
    length = 24
    boundary_buffer = 2
    trajectory = _manager(
        length,
        num_steps=100,
        extruder_count=3,
        off_rate=0.15,
        seed=2026,
        boundary_buffer=boundary_buffer,
    ).simulate()

    assert trajectory.shape == (101, 3, 2)
    assert trajectory[..., 0].min() >= boundary_buffer
    assert trajectory[..., 1].max() <= length - boundary_buffer - 1
    assert np.all(trajectory[..., 0] < trajectory[..., 1])
    for frame in trajectory:
        assert np.unique(frame).size == frame.size
        _assert_no_interleaving([tuple(map(int, pair)) for pair in frame])


def test_unfix_conversion_factor_is_honored_in_probabilities_and_dynamics():
    fixation = np.ones(20)
    fixed = LoopExtrusionManager(
        fixation,
        fixation,
        num_steps=4,
        extruder_count=1,
        off_rate=0,
        unfix_conversion_factor=0,
        seed=11,
    )
    immediately_unfixed = LoopExtrusionManager(
        fixation,
        fixation,
        num_steps=4,
        extruder_count=1,
        off_rate=0,
        unfix_conversion_factor=1,
        seed=11,
    )

    np.testing.assert_array_equal(fixed.fprobs_unfix, np.zeros(20))
    np.testing.assert_array_equal(immediately_unfixed.fprobs_unfix, np.ones(20))
    np.testing.assert_array_equal(
        fixed.simulate(),
        np.array([[[3, 5]]] * 5, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        immediately_unfixed.simulate(),
        np.array(
                [
                    [[3, 5]],
                    [[2, 6]],
                    [[4, 6]],
                    [[3, 7]],
                    [[2, 8]],
                ],
            dtype=np.int64,
        ),
    )


def test_unfix_probabilities_are_zero_for_zero_fixation_and_clipped_at_one():
    fixation = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 0.0, 0.2, 0.8])
    manager = LoopExtrusionManager(
        fixation,
        fixation,
        num_steps=0,
        extruder_count=1,
        unfix_conversion_factor=0.2,
        seed=1,
    )

    np.testing.assert_allclose(
        manager.fprobs_unfix,
        [0.0, 1.0, 0.8, 0.4, 0.2, 0.0, 1.0, 0.25],
        rtol=0,
        atol=0,
    )


def test_overcapacity_fails_during_construction_instead_of_searching_forever():
    with pytest.raises(ValueError, match="placement capacity 1"):
        LoopExtrusionManager(
            np.zeros(8),
            np.zeros(8),
            num_steps=1,
            extruder_count=2,
            seed=1,
        )


def test_minimum_chromosome_accepts_its_single_legal_loop_pair():
    trajectory = LoopExtrusionManager(
        np.zeros(7),
        np.zeros(7),
        num_steps=0,
        extruder_count=1,
        boundary_buffer=2,
        seed=1,
    ).simulate()

    np.testing.assert_array_equal(trajectory, [[[2, 4]]])
    with pytest.raises(ValueError, match="chromosome is too short"):
        LoopExtrusionManager(
            np.zeros(6),
            np.zeros(6),
            num_steps=0,
            extruder_count=1,
            boundary_buffer=2,
        )


def test_exact_declared_capacity_is_placeable_for_every_seed():
    # Length 10 with the default two-bead boundary has four possible centers;
    # centers 3 and 6 provide the declared capacity of two extruders.  A random
    # greedy first choice must not turn that valid configuration into a false
    # placement error.
    for seed in range(10):
        trajectory = LoopExtrusionManager(
            np.zeros(10),
            np.zeros(10),
            num_steps=0,
            extruder_count=2,
            seed=seed,
        ).simulate()
        assert trajectory.shape == (1, 2, 2)
        assert np.unique(trajectory[0]).size == trajectory[0].size
        _assert_no_interleaving(
            [tuple(map(int, pair)) for pair in trajectory[0]]
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 19])
def test_dense_boundary_reloads_complete_without_transient_placement_failure(seed):
    length = 10
    trajectory = LoopExtrusionManager(
        np.zeros(length),
        np.zeros(length),
        num_steps=30,
        extruder_count=2,
        off_rate=0.0,
        unfix_conversion_factor=0.0,
        seed=seed,
    ).simulate()

    assert trajectory.shape == (31, 2, 2)
    assert trajectory[..., 0].min() >= 2
    assert trajectory[..., 1].max() <= length - 3
    for frame in trajectory:
        assert np.unique(frame).size == frame.size
        assert np.all(frame[:, 0] < frame[:, 1])
        _assert_no_interleaving(
            [tuple(map(int, pair)) for pair in frame]
        )


def test_collective_reload_policy_completes_blocked_dense_state_and_counts_use():
    manager, initial = _manager_with_blocked_local_reload()

    trajectory = manager.simulate()

    np.testing.assert_array_equal(trajectory[0], initial)
    assert trajectory.shape == (2, 2, 2)
    assert manager.reload_policy == "collective"
    assert manager.collective_reload_count == 1
    for frame in trajectory:
        assert np.unique(frame).size == frame.size
        assert np.all(frame[:, 0] < frame[:, 1])
        _assert_no_interleaving(
            [tuple(map(int, pair)) for pair in frame]
        )


def test_error_reload_policy_reports_blocked_dense_state():
    manager, _ = _manager_with_blocked_local_reload(reload_policy="error")

    with pytest.raises(RuntimeError, match="no legal reload sites remain"):
        manager.simulate()

    assert manager.collective_reload_count == 0


@pytest.mark.parametrize(
    ("reload_policy", "error", "message"),
    [
        (None, TypeError, "reload_policy must be a string"),
        ("local", ValueError, "reload_policy must be 'collective' or 'error'"),
    ],
)
def test_invalid_reload_policy_is_rejected(reload_policy, error, message):
    with pytest.raises(error, match=message):
        _manager(reload_policy=reload_policy)


@pytest.mark.parametrize(
    ("forward", "reverse", "error", "message"),
    [
        (np.zeros((2, 4)), np.zeros(8), ValueError, "one-dimensional"),
        (np.array([]), np.array([]), ValueError, "must not be empty"),
        (["x"] * 8, np.zeros(8), TypeError, "numeric"),
        ([0, 0, 0, np.nan, 0, 0, 0, 0], np.zeros(8), ValueError, "finite"),
        ([0, 0, 0, -0.1, 0, 0, 0, 0], np.zeros(8), ValueError, "between 0 and 1"),
        ([0, 0, 0, 1.1, 0, 0, 0, 0], np.zeros(8), ValueError, "between 0 and 1"),
        (np.zeros(8), np.zeros(9), ValueError, "same length"),
    ],
)
def test_invalid_fixation_vectors_are_rejected(
    forward, reverse, error, message
):
    with pytest.raises(error, match=message):
        LoopExtrusionManager(forward, reverse, num_steps=1, extruder_count=1)


def test_complex_fixation_probabilities_are_rejected_instead_of_truncated():
    with pytest.raises(TypeError):
        LoopExtrusionManager(
            np.full(8, 0.25 + 0.5j),
            np.zeros(8),
            num_steps=1,
            extruder_count=1,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"num_steps": True}, TypeError, "num_steps must be an integer"),
        ({"num_steps": -1}, ValueError, "num_steps must be at least 0"),
        ({"extruder_count": 0}, ValueError, "extruder_count must be at least 1"),
        ({"extruder_count": 1.5}, TypeError, "extruder_count must be an integer"),
        ({"off_rate": True}, TypeError, "off_rate must be a real number"),
        ({"off_rate": -0.1}, ValueError, "off_rate must be between 0 and 1"),
        ({"off_rate": 1.1}, ValueError, "off_rate must be between 0 and 1"),
        (
            {"unfix_conversion_factor": -0.1},
            ValueError,
            "unfix_conversion_factor must be finite and non-negative",
        ),
        (
            {"unfix_conversion_factor": "x"},
            TypeError,
            "unfix_conversion_factor must be a real number",
        ),
        ({"boundary_buffer": 0}, ValueError, "boundary_buffer must be at least 1"),
        ({"boundary_buffer": 9}, ValueError, "chromosome is too short"),
        ({"seed": 1.5}, TypeError, "seed must be an integer or None"),
        ({"seed": True}, TypeError, "seed must be an integer or None"),
    ],
)
def test_invalid_manager_options_are_rejected(overrides, error, message):
    with pytest.raises(error, match=message):
        _manager(**overrides)


def test_loop_bond_updater_normalizes_pairs_and_reports_active_union():
    trajectory = np.array(
        [
            [[2, 0], [5, 3]],
            [[3, 0], [5, 2]],
            [[0, 2], [2, 5]],
        ]
    )
    updater = LoopBondUpdater(trajectory, k_loop=2.0, r0_loop=1.0)

    assert updater.pairs == ((0, 2), (0, 3), (2, 5), (3, 5))
    assert updater.active_pairs(0) == ((0, 2), (3, 5))
    assert updater.active_pairs(1) == ((0, 3), (2, 5))
    assert updater.active_pairs(2) == ((0, 2), (2, 5))
    assert updater.particle_count == 6
    assert updater.current_step is None


def test_two_dimensional_loop_frame_is_accepted_as_one_frame():
    updater = LoopBondUpdater([[3, 1], [4, 6]])

    assert updater.trajectory.shape == (1, 2, 2)
    assert updater.active_pairs(0) == ((1, 3), (4, 6))


@pytest.mark.parametrize(
    ("trajectory", "error", "message"),
    [
        ([0, 1], ValueError, "shape"),
        (np.empty((0, 1, 2), dtype=int), ValueError, "at least one frame"),
        (np.empty((1, 0, 2), dtype=int), ValueError, "at least one frame"),
        ([[["a", "b"]]], TypeError, "integer particle indices"),
        ([[[0.0, 1.5]]], TypeError, "integer particle indices"),
        ([[[0.0, np.nan]]], TypeError, "integer particle indices"),
        ([[[-1, 2]]], ValueError, "non-negative"),
        ([[[2, 2]]], ValueError, "cannot connect a particle to itself"),
        ([[[0, 2], [2, 0]]], ValueError, "duplicate loop bonds"),
    ],
)
def test_invalid_loop_trajectories_are_rejected(trajectory, error, message):
    with pytest.raises(error, match=message):
        LoopBondUpdater(trajectory)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"k_loop": -1}, ValueError, "k_loop must be finite and non-negative"),
        ({"k_loop": True}, TypeError, "k_loop must be a real number"),
        ({"r0_loop": np.inf}, ValueError, "r0_loop must be finite and non-negative"),
        ({"r0_loop": "x"}, TypeError, "r0_loop must be a real number"),
    ],
)
def test_invalid_loop_force_options_are_rejected(kwargs, error, message):
    with pytest.raises(error, match=message):
        LoopBondUpdater([[[0, 2]]], **kwargs)


def test_active_pair_step_validation_is_explicit():
    updater = LoopBondUpdater([[[0, 2]], [[1, 3]]])

    with pytest.raises(TypeError, match="step must be an integer"):
        updater.active_pairs(True)
    with pytest.raises(ValueError, match="step must be at least 0"):
        updater.active_pairs(-1)
    with pytest.raises(IndexError, match="outside a 2-frame trajectory"):
        updater.active_pairs(2)


def test_openmm_force_updates_live_context_parameters_and_energy():
    trajectory = np.array(
        [
            [[0, 2], [3, 5]],
            [[0, 3], [2, 5]],
            [[2, 0], [5, 2]],
        ]
    )
    updater = LoopBondUpdater(trajectory, k_loop=2.0, r0_loop=1.0)
    force = updater.create_force(openmm)

    assert force.getName() == "LoopExtrusion"
    assert _bond_parameters(force) == [
        (0, 2, 1.0),
        (0, 3, 0.0),
        (2, 5, 0.0),
        (3, 5, 1.0),
    ]
    assert updater.current_step == 0

    system = openmm.System()
    for _ in range(updater.particle_count):
        system.addParticle(1.0)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(
        [openmm.Vec3(index, 0, 0) for index in range(updater.particle_count)]
        * unit.nanometer
    )

    try:
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
        updater.set_step(1, context)
        updated_energy = context.getState(getEnergy=True).getPotentialEnergy()

        assert initial_energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(2.0)
        assert updated_energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(8.0)
        assert _bond_parameters(force) == [
            (0, 2, 0.0),
            (0, 3, 1.0),
            (2, 5, 1.0),
            (3, 5, 0.0),
        ]
        assert updater.current_step == 1
    finally:
        del context
        del integrator


def test_force_lifecycle_validation_prevents_invalid_updates():
    updater = LoopBondUpdater([[[0, 2]], [[1, 3]]])

    with pytest.raises(RuntimeError, match="create_force"):
        updater.set_step(1, None)
    updater.create_force(openmm)
    with pytest.raises(RuntimeError, match="already been created"):
        updater.create_force(openmm)


def test_differently_named_dynamic_loop_forces_share_one_context():
    first_trajectory = np.array([[[0, 2]], [[0, 3]]])
    second_trajectory = np.array([[[3, 5]], [[2, 5]]])
    simulation = MiChroM(verbose=False)
    simulation.mm = openmm
    simulation.N = 6
    simulation.forceDict = {}

    first = simulation.addDynamicLoopPotential(
        first_trajectory,
        k_loop=2.0,
        r0_loop=1.0,
        name="EnhancerLoops",
    )
    second = simulation.addDynamicLoopPotential(
        second_trajectory,
        k_loop=4.0,
        r0_loop=0.5,
        name="BoundaryLoops",
    )

    assert first.force.getName() == "EnhancerLoops"
    assert second.force.getName() == "BoundaryLoops"
    assert set(simulation.forceDict) == {"EnhancerLoops", "BoundaryLoops"}

    system = openmm.System()
    for _ in range(simulation.N):
        system.addParticle(1.0)
    for force in simulation.forceDict.values():
        system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(
        [openmm.Vec3(index, 0, 0) for index in range(simulation.N)]
        * unit.nanometer
    )
    simulation.context = context

    try:
        initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
        simulation.updateDynamicLoopPotential(1, name="EnhancerLoops")
        simulation.updateDynamicLoopPotential(1, name="BoundaryLoops")
        updated_energy = context.getState(getEnergy=True).getPotentialEnergy()

        assert initial_energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(5.5)
        assert updated_energy.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(16.5)
        assert first.current_step == 1
        assert second.current_step == 1
    finally:
        del simulation.context
        del context
        del integrator


def test_dynamic_loop_force_cannot_be_added_after_simulation_creation(tmp_path):
    simulation = MiChroM(verbose=False)
    simulation.setup(platform="CPU", verbose=False)
    simulation.saveFolder(tmp_path)
    simulation.loadStructure(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        center=False,
    )
    simulation.createSimulation(verbose=False)

    try:
        with pytest.raises(RuntimeError, match="before createSimulation"):
            simulation.addDynamicLoopPotential([[[0, 2]]], name="TooLate")
    finally:
        del simulation.context
        del simulation.simulation
        del simulation.integrator


def test_loop_extrusion_api_and_legacy_spelling_are_true_package_aliases():
    assert Loop_Extrusion_Manager is LoopExtrusionManager
    assert OpenMiChroM.LoopExtrusionManager is LoopExtrusionManager
    assert OpenMiChroM.Loop_Extrusion_Manager is LoopExtrusionManager
    assert OpenMiChroM.LoopBondUpdater is LoopBondUpdater
