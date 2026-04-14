# DroneRacingEnv is imported lazily to avoid pulling in gym-pybullet-drones
# (and its transitive deps like transforms3d) when only gate/reward data is needed.
from .gate_manager import GateManager, Gate, RACE_GATES
from .reward import RewardComputer

__all__ = ["DroneRacingEnv", "GateManager", "Gate", "RACE_GATES", "RewardComputer"]


def __getattr__(name: str):
    if name == "DroneRacingEnv":
        from .drone_racing_env import DroneRacingEnv
        return DroneRacingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
