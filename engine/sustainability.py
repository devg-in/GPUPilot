"""Sustainability and impact calculations for GPUPilot.

All environmental values are estimates derived from simulated IT power and
configurable assumptions. They are not measurements of a real data center.
"""
from dataclasses import dataclass, asdict
from typing import Dict

from backend.schemas.models import SteadyStateEstimate


@dataclass(frozen=True)
class EnvironmentalAssumptions:
    # Facility overhead: total facility energy / IT energy.
    pue: float = 1.20
    # Liters of water per kWh of total facility energy. Configurable assumption.
    wue_l_per_kwh: float = 0.50


DEFAULT_ENV = EnvironmentalAssumptions()


def _water_l_per_hour(power_w: float, env: EnvironmentalAssumptions) -> float:
    facility_kwh = max(power_w, 0.0) * env.pue / 1000.0
    return facility_kwh * env.wue_l_per_kwh


def estimate_impact(estimate: SteadyStateEstimate, env: EnvironmentalAssumptions = DEFAULT_ENV) -> Dict[str, float | str]:
    power_w = float(estimate.total_power)
    energy_kwh_per_hour = power_w / 1000.0
    facility_energy_kwh_per_hour = energy_kwh_per_hour * env.pue
    cooling_energy_kwh_per_hour = max(facility_energy_kwh_per_hour - energy_kwh_per_hour, 0.0)
    water_l_per_hour = _water_l_per_hour(power_w, env)
    thermal_risk = "LOW" if estimate.temperature < 65 else "MEDIUM" if estimate.temperature < 78 else "HIGH"
    return {
        "data_source": "SIMULATED GPU TELEMETRY",
        "power_w": round(power_w, 2),
        "energy_kwh_per_hour": round(energy_kwh_per_hour, 4),
        "facility_energy_kwh_per_hour": round(facility_energy_kwh_per_hour, 4),
        "estimated_cooling_energy_kwh_per_hour": round(cooling_energy_kwh_per_hour, 4),
        "estimated_water_l_per_hour": round(water_l_per_hour, 4),
        "temperature_c": round(float(estimate.temperature), 2),
        "thermal_risk": thermal_risk,
        "latency_ms": round(float(estimate.latency), 2),
        "throughput_req_min": round(float(estimate.throughput), 2),
        "sla_status": estimate.sla.sla_status,
        "pue_assumption": env.pue,
        "wue_assumption_l_per_kwh": env.wue_l_per_kwh,
    }


def compare_impact(before: SteadyStateEstimate, after: SteadyStateEstimate,
                   env: EnvironmentalAssumptions = DEFAULT_ENV) -> Dict[str, object]:
    b = estimate_impact(before, env)
    a = estimate_impact(after, env)

    def pct(old: float, new: float) -> float:
        return round(((old - new) / old) * 100.0, 2) if old else 0.0

    return {
        "before": b,
        "after": a,
        "changes": {
            "power_saved_w": round(float(b["power_w"]) - float(a["power_w"]), 2),
            "power_reduction_pct": pct(float(b["power_w"]), float(a["power_w"])),
            "energy_saved_kwh_per_hour": round(float(b["energy_kwh_per_hour"]) - float(a["energy_kwh_per_hour"]), 4),
            "temperature_reduction_c": round(float(b["temperature_c"]) - float(a["temperature_c"]), 2),
            "water_reduction_l_per_hour": round(float(b["estimated_water_l_per_hour"]) - float(a["estimated_water_l_per_hour"]), 4),
            "water_reduction_pct": pct(float(b["estimated_water_l_per_hour"]), float(a["estimated_water_l_per_hour"])),
            "latency_change_ms": round(float(a["latency_ms"]) - float(b["latency_ms"]), 2),
            "throughput_change_req_min": round(float(a["throughput_req_min"]) - float(b["throughput_req_min"]), 2),
        },
        "assumptions": asdict(env),
    }
