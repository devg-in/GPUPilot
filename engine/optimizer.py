from dataclasses import dataclass
from typing import List, Tuple
from itertools import product

from backend.schemas.models import Configuration, SteadyStateEstimate
from simulation.simulator import GPUSimulator
from engine.sustainability import DEFAULT_ENV, estimate_impact


@dataclass
class OptimizationResult:
    current_configuration: Configuration
    recommended_configuration: Configuration
    current_estimate: SteadyStateEstimate
    recommended_estimate: SteadyStateEstimate
    candidates_checked: int
    feasible_candidates: int
    reason: str


class GPUPilotOptimizer:
    """SLA-constrained, multi-objective optimizer for the simulator."""

    def __init__(self, simulator: GPUSimulator):
        self.simulator = simulator
        self.last_result = None
        self.last_evaluated_at = None

    def generate_candidates(self) -> List[Configuration]:
        """Generate a broad, deduplicated search space across all tunable knobs.

        Every combination is evaluated by the same SLA/quality feasibility path;
        candidates are never applied merely to make the dashboard look busy.
        """
        current = self.simulator.configuration
        batches = [8, 16, 32, 64]
        precisions = ["FP32", "FP16", "INT8"]
        gpus = [1, 2, 3, 4, 5]
        replicas = [1, 2, 3]
        concurrencies = [16, 24, 32, 64]

        candidates = [current]
        for precision, batch, gpu, replica, concurrency in product(
            precisions, batches, gpus, replicas, concurrencies
        ):
            # The hackathon simulator models a five-GPU cluster. GPU allocation
            # is per replica, so use the total footprint for all safety checks.
            if gpu * replica > 5:
                continue
            candidates.append(current.model_copy(update={
                "precision": precision,
                "batch_size": batch,
                "gpu_allocation": gpu,
                "replicas": replica,
                "concurrency": concurrency,
            }))

        unique = {}
        for candidate in candidates:
            key = (candidate.gpu_allocation, candidate.batch_size, candidate.precision,
                   candidate.replicas, candidate.concurrency)
            unique[key] = candidate
        return list(unique.values())

    def required_gpu_count(self, request_rate: float) -> int:
        """Minimum simulated GPU count required by the workload demand.

        This mirrors the cluster scheduler's demand floor so the optimizer and
        dashboard cannot disagree about how many GPUs are actually required.
        """
        return max(1, min(5, int((float(request_rate) / 900.0) + 0.999)))

    @staticmethod
    def sustainability_score(estimate: SteadyStateEstimate) -> float:
        """Lower is better; all candidates reaching this function passed SLA."""
        impact = estimate_impact(estimate, DEFAULT_ENV)
        # Normalize dimensions to comparable prototype-scale values.
        power = float(impact["power_w"]) / 250.0
        thermal = float(estimate.temperature) / 70.0
        water = float(impact["estimated_water_l_per_hour"]) / 0.2
        memory = float(estimate.total_memory) / 16.0
        return 0.55 * power + 0.20 * thermal + 0.15 * water + 0.10 * memory

    def optimize(self) -> OptimizationResult:
        current = self.simulator.configuration
        request_rate = self.simulator.latest.request_rate
        required_gpus = self.required_gpu_count(request_rate)
        current_estimate = self.simulator.estimate_steady_state(current, request_rate)
        candidates = self.generate_candidates()
        feasible: List[Tuple[float, Configuration, SteadyStateEstimate]] = []

        for candidate in candidates:
            # Never recommend fewer GPUs than the workload floor. This keeps
            # optimizer state aligned with the dynamic cluster scheduler.
            if candidate.gpu_allocation * candidate.replicas < required_gpus:
                continue
            estimate = self.simulator.estimate_steady_state(candidate, request_rate)
            if estimate.feasible:
                feasible.append((self.sustainability_score(estimate), candidate, estimate))

        if not feasible:
            result = OptimizationResult(current, current, current_estimate, current_estimate,
                                        len(candidates), 0,
                                        f"No feasible configuration satisfies the SLA for the required {required_gpus} GPU workload floor.")
            from datetime import datetime, timezone
            self.last_result = result
            self.last_evaluated_at = datetime.now(timezone.utc).isoformat()
            return result

        # Choose from measured model estimates only. Workload demand already
        # influences capacity, latency, throughput, utilization, and SLA feasibility;
        # avoid fixed request-rate bands that force cosmetic configuration changes.
        feasible.sort(key=lambda x: (x[0], x[2].total_power, x[2].temperature))
        _, best_config, best_estimate = feasible[0]

        if best_config == current:
            reason = "Current configuration is already the best feasible operating point for the current workload."
        else:
            changes = []
            for field in ["precision", "batch_size", "gpu_allocation", "replicas", "concurrency"]:
                old, new = getattr(current, field), getattr(best_config, field)
                if old != new:
                    changes.append(f"{field} {old} → {new}")
            reason = (f"Changed {', '.join(changes)} because the workload currently requires at least "
                      f"{required_gpus} GPU(s) and the selected configuration remains within SLA "
                      "while minimizing estimated sustainability impact.")

        result = OptimizationResult(current, best_config, current_estimate, best_estimate,
                                    len(candidates), len(feasible), reason)
        from datetime import datetime, timezone
        self.last_result = result
        self.last_evaluated_at = datetime.now(timezone.utc).isoformat()
        return result
