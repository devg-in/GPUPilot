"""Telemetry provider abstraction.

The hackathon uses the simulator provider. NVML/DCGM providers can be added
without changing the controller contract.
"""
from abc import ABC, abstractmethod


class TelemetryProvider(ABC):
    @abstractmethod
    def latest(self):
        raise NotImplementedError

    @abstractmethod
    def history(self, limit=100):
        raise NotImplementedError


class SimulatorTelemetryProvider(TelemetryProvider):
    def __init__(self, simulator):
        self.simulator = simulator

    @property
    def latest(self):
        return self.simulator.latest

    def history(self, limit=100):
        return self.simulator.get_history(include_live=True, limit=limit)
