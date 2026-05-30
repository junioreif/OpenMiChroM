__version__ = '1.1.1'

from .CndbTools import CndbTools, cndbTools

__all__ = [
    "ActiveBrownianIntegrator",
    "CndbTools",
    "CustomMiChroMTraining",
    "FullTraining",
    "MiChroM",
    "SaveStructure",
    "SimulationReporter",
    "cndbTools",
]


def __getattr__(name):
    """Lazily import OpenMM-dependent objects.

    This keeps ``from OpenMiChroM.CndbTools import CndbTools`` usable in
    analysis-only environments that do not have OpenMM installed.
    """

    if name == "MiChroM":
        from .ChromDynamics import MiChroM

        return MiChroM
    if name in {"FullTraining", "CustomMiChroMTraining"}:
        from .Optimization import FullTraining, CustomMiChroMTraining

        return {
            "FullTraining": FullTraining,
            "CustomMiChroMTraining": CustomMiChroMTraining,
        }[name]
    if name == "ActiveBrownianIntegrator":
        from .Integrators import ActiveBrownianIntegrator

        return ActiveBrownianIntegrator
    if name in {"SaveStructure", "SimulationReporter"}:
        from .CustomReporter import SaveStructure, SimulationReporter

        return {
            "SaveStructure": SaveStructure,
            "SimulationReporter": SimulationReporter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
