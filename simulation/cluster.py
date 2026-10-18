"""Dynamic 5-GPU simulated cluster state for the GPUPilot dashboard."""
from typing import Any, Dict, List

GPU_SPECS = [
    {"id": "GPU 01", "model": "RTX 4090", "capacity": 1.00, "power_bias": 1.00, "thermal_bias": 1.00},
    {"id": "GPU 02", "model": "A100", "capacity": 1.35, "power_bias": 0.92, "thermal_bias": 0.96},
    {"id": "GPU 03", "model": "A100", "capacity": 1.35, "power_bias": 0.94, "thermal_bias": 1.00},
    {"id": "GPU 04", "model": "L40S", "capacity": 1.15, "power_bias": 0.88, "thermal_bias": 0.94},
    {"id": "GPU 05", "model": "H100", "capacity": 1.65, "power_bias": 0.86, "thermal_bias": 0.96},
]


def build_cluster(sim) -> List[Dict[str, Any]]:
    t = sim.latest
    cfg = sim.configuration
    demand = float(t.request_rate)
    # The simulator configuration is the single source of truth for actual
    # allocation. The optimizer/controller must explicitly change it; the
    # dashboard must never display extra GPUs as active before they are
    # actually allocated.
    required = max(1, min(5, int((demand / 900.0) + 0.999)))
    configured = max(1, min(5, cfg.gpu_allocation * cfg.replicas))
    active_count = configured

    # Choose GPUs using current thermal/utilization/power state and capacity.
    scored = []
    base_util = float(t.gpu_utilization)
    base_temp = float(t.temperature)
    base_power = float(t.gpu_power)
    for spec in GPU_SPECS:
        # Lower score is better. Capacity helps at high demand; thermal/power stress is penalized.
        score = (base_temp * spec["thermal_bias"] * 0.55 + base_power * spec["power_bias"] * 0.12
                 - spec["capacity"] * min(demand / 1000.0, 1.5) * 18.0)
        scored.append((score, spec))
    scored.sort(key=lambda x: x[0])
    selected = {spec["id"] for _, spec in scored[:active_count]}

    rows = []
    for idx, spec in enumerate(GPU_SPECS):
        active = spec["id"] in selected
        if active:
            # Spread the aggregate telemetry across selected GPUs with small deterministic differences.
            factor = 0.94 + ((idx * 7) % 9) / 100.0
            util = min(98.0, max(12.0, base_util * factor))
            temp = max(42.0, base_temp * (0.96 + (idx % 5) * 0.012))
            power = max(80.0, base_power * spec["power_bias"] * (0.95 + (idx % 4) * 0.018))
            memory = max(1.0, float(t.memory) * (0.90 + (idx % 4) * 0.035))
            status = "Overloaded" if util >= 90.0 else "Active"
        else:
            util = 8.0 + (idx * 3.0) % 8.0
            temp = max(42.0, 43.0 + idx * 1.4)
            power = 55.0 + idx * 7.0
            memory = 1.5 + idx * 0.25
            status = "Idle"
        rows.append({
            "id": spec["id"], "model": spec["model"], "capacity_factor": spec["capacity"],
            "utilization": round(util, 1), "temperature": round(temp, 1), "power": round(power, 1),
            "memory": round(memory, 2), "memory_capacity": 24.0, "status": status,
            "active": active, "workload": "LLM Inference" if active else "—",
        })
    return rows
