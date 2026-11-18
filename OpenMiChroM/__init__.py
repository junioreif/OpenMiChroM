__version__ = '2.0.0rc1'

from importlib import import_module

from .CndbTools import CndbTools, cndbTools


_LAZY_EXPORTS = {
    "MiChroM": ".ChromDynamics",
    "FullTraining": ".Optimization",
    "CustomMiChroMTraining": ".Optimization",
    "ActiveBrownianIntegrator": ".Integrators",
    "SaveStructure": ".CustomReporter",
    "SimulationReporter": ".CustomReporter",
    "convert": ".Converters",
    "ndb_to_cndb": ".Converters",
    "cndb_to_ndb": ".Converters",
    "ndb_to_pdb": ".Converters",
    "pdb_to_ndb": ".Converters",
    "ndb_to_spw": ".Converters",
    "spw_to_ndb": ".Converters",
    "gro_to_ndb": ".Converters",
    "csv_to_ndb": ".Converters",
    "StructuralVariantResult": ".StructuralVariants",
    "add_ideal_chromosome": ".StructuralVariants",
    "apply_structural_variant": ".StructuralVariants",
    "delete_region": ".StructuralVariants",
    "duplicate_region": ".StructuralVariants",
    "ideal_chromosome_profile": ".StructuralVariants",
    "invert_region": ".StructuralVariants",
    "locus_labels": ".StructuralVariants",
    "read_locus_matrix": ".StructuralVariants",
    "remove_ideal_chromosome": ".StructuralVariants",
    "write_locus_matrix": ".StructuralVariants",
    "write_locus_sequence": ".StructuralVariants",
    "LoopBondUpdater": ".Extrusion_Bonds",
    "LoopExtrusionManager": ".Extrusion_Bonds",
    "Loop_Extrusion_Manager": ".Extrusion_Bonds",
}


__all__ = ["CndbTools", "cndbTools", *_LAZY_EXPORTS]


def __getattr__(name):
    """Load optional public helpers lazily."""

    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
