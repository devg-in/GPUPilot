"""
config.py - constants for the GPUPilot GPU simulator.

*** SIMULATED GPU TELEMETRY *** - nothing here describes a real NVIDIA GPU.
Every constant is calibrated so the DEFAULT configuration at ~850 req/min lands
on the baseline: util 82 %, power 285 W, temp 73 C, mem 14 GB, latency 165 ms,
throughput ~820 req/min.
"""
from dataclasses import dataclass

DATA_SOURCE_LABEL = "SIMULATED GPU TELEMETRY"

SEED = 42                  # reproducible seed used by the simulator CLI; app runs randomize by default
INITIAL_SAMPLES = 100
SAMPLE_INTERVAL_S = 1.0    # one live tick per second (also spacing of the backdated initial samples)
HISTORY_MAXLEN = 5000

# (min, max) simulation bounds - values are clamped to these.
BOUNDS = {
    "gpu_utilization": (20.0, 98.0),
    "gpu_power": (120.0, 350.0),
    "temperature": (45.0, 88.0),
    "memory": (6.0, 24.0),
    "request_rate": (300.0, 2200.0),
    "latency": (60.0, 300.0),
    "throughput": (250.0, 1200.0),
}

BASELINE = {
    "gpu_utilization": 82.0, "gpu_power": 285.0, "temperature": 73.0,
    "memory": 14.0, "request_rate": 850.0, "latency": 165.0, "throughput": 820.0,
}

DEFAULT_CONFIGURATION = {
    "gpu_allocation": 1, "batch_size": 16, "precision": "FP16",
    "replicas": 1, "concurrency": 32,
}

DEFAULT_SLA = {
    "max_latency_ms": 200.0,
    "min_throughput_req_min": 700.0,
    "max_temperature_c": 85.0,
}

# ---------------------------------------------------------------- workload
NORMAL_RATE = 850.0        # req/min in NORMAL state
DEFAULT_PEAK_RATE = 1150.0
SPIKE_TAU = 5.0            # spike ramp time-constant (ticks)
RECOVERY_TAU = 6.0         # recovery decay time-constant (ticks)
DEFAULT_SPIKE_TICKS = 5
DEFAULT_PEAK_HOLD_TICKS = 5
SCRIPTED_SPIKE_AT = 31
SCRIPTED_PEAK_HOLD = DEFAULT_PEAK_HOLD_TICKS

# Live demo automation: after a stable NORMAL period, the simulator
# automatically injects a workload spike. This keeps the demo fully autonomous.
AUTO_SPIKE_ENABLED = True
AUTO_SPIKE_INTERVAL_TICKS = 20
AUTO_SPIKE_PEAK_RATES = [980.0, 1120.0, 1280.0, 1450.0, 1750.0, 2050.0, 1350.0]
AUTO_SPIKE_PEAK_RATE = DEFAULT_PEAK_RATE
AUTO_SPIKE_DURATION_TICKS = DEFAULT_SPIKE_TICKS
AUTO_SPIKE_PEAK_HOLD_TICKS = DEFAULT_PEAK_HOLD_TICKS

# ------------------------------------------------- capacity / queueing model
BASE_CAPACITY = 1000.0     # req/min the default config can serve
BASE_LATENCY_MS = 95.0     # unloaded service latency of default config
QUEUE_A = 1.41             # latency = L0 * (1 + A * rho^4)
SOFTMIN_P = 8              # sharpness of the throughput saturation knee

# ---------------------------------------------------- power / thermal model
IDLE_POWER_W = 90.0
DYNAMIC_POWER_W = 195.0    # => 90 + 195 = 285 W at 82 % util (FP16)
POWER_EXPONENT = 1.3       # mildly super-linear power vs utilization
AMBIENT_C = 34.0
THERMAL_RESISTANCE = 0.137 # C per W => 34 + 0.137*285 = 73.0 C
THERMAL_SMOOTHING = 0.85   # temp[t] = 0.85*temp[t-1] + 0.15*target

# ----------------------------------------------- latent-state tracking rates
REQ_TRACKING = 0.82
UTIL_TRACKING = 0.58
MEM_TRACKING = 0.25
LAT_TRACKING = 0.40
THR_TRACKING = 0.50


@dataclass(frozen=True)
class PrecisionProfile:
    speed: float       # relative serving speed
    power: float       # multiplier on the dynamic power term
    memory: float      # multiplier on memory footprint


PRECISION = {
    "FP32": PrecisionProfile(speed=0.55, power=1.15, memory=2.00),
    "FP16": PrecisionProfile(speed=1.00, power=1.00, memory=1.00),
    "INT8": PrecisionProfile(speed=1.45, power=0.88, memory=0.55),
}
