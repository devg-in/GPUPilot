"""
demo_terminal.py - run the GPU simulator and print everything in the terminal.
SIMULATED GPU TELEMETRY - no server, no browser needed.

    python demo_terminal.py          # every 5th sample of the 100
    python demo_terminal.py --all    # all 100 samples
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sys
from backend.schemas.models import ConfigurationUpdate
from simulation.simulator import GPUSimulator

HEAD = f"{'id':>3} {'state':<9} {'req/min':>7} {'util%':>6} {'power W':>8} {'temp C':>7} {'mem GB':>7} {'lat ms':>7} {'thr/min':>8}  SLA"


def row(s):
    bar = "#" * int(s.gpu_utilization / 5)          # 1 '#' = 5 % utilization
    return (f"{s.sample_id:>3} {s.workload_state.value:<9} {s.request_rate:>7} {s.gpu_utilization:>6.1f} "
            f"{s.gpu_power:>8.1f} {s.temperature:>7.1f} {s.memory:>7.2f} {s.latency:>7.1f} "
            f"{s.throughput:>8}  {s.sla_status:<4} {bar}")


def main():
    show_all = "--all" in sys.argv
    sim = GPUSimulator()

    print("=" * 100)
    print("PART 1 - the initial 100 samples        *** SIMULATED GPU TELEMETRY ***")
    print("  1-30 normal | 31-50 spike | 51-75 peak | 76-100 recovery -> normal")
    print("=" * 100)
    print(HEAD)
    for s in sim.get_history():
        if show_all or s.sample_id == 1 or s.sample_id % 5 == 0:
            print(row(s))

    print()
    print("=" * 100)
    print("PART 2 - live demo: spike, then GPUPilot changes the configuration")
    print("=" * 100)
    sim.reset()
    print("\n[1] Normal operation (default config):")
    print(HEAD)
    for s in sim.step(3):
        print(row(s))

    print("\n[2] SPIKE triggered! Watch request rate, util, power rise, then latency break the 200 ms SLA:")
    print(HEAD)
    sim.trigger_spike()
    for i, s in enumerate(sim.step(40), 1):
        if i % 4 == 0:
            print(row(s))

    print(f"\n[3] GPUPilot applies a new config: precision=INT8, batch_size=32  (was FP16 / 16)")
    sim.apply_configuration(ConfigurationUpdate(precision="INT8", batch_size=32))
    print(HEAD)
    for i, s in enumerate(sim.step(24), 1):
        if i % 3 == 0:
            print(row(s))

    before, after, delta, status = sim.comparison()
    print("\n[4] BEFORE vs AFTER the config change (average of 5 samples each):")
    print(f"    {'metric':<16}{'before':>10}{'after':>10}{'change':>10}")
    for k in ["request_rate", "gpu_utilization", "gpu_power", "temperature", "memory", "latency", "throughput"]:
        print(f"    {k:<16}{before[k]:>10.1f}{after[k]:>10.1f}{delta[k]:>+10.1f}")
    print(f"    SLA after change: {status}")
    print("\nAll values above are SIMULATED GPU TELEMETRY (not from a real NVIDIA GPU).")


if __name__ == "__main__":
    main()
