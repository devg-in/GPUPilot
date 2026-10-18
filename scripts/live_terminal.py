"""
live_terminal.py - LIVE view of the simulator: a new reading every second.
*** SIMULATED GPU TELEMETRY *** (not from a real NVIDIA GPU)

    python live_terminal.py            # hands-free demo: normal -> spike -> GPUPilot fixes it
    python live_terminal.py --manual   # you control it with keys (below)
    python live_terminal.py --interval 0.5

Keys (Windows terminal), work in both modes:
    s = trigger workload spike      o = apply optimized config (INT8, batch 32)
    d = back to default config      r = start recovery          q = quit
(Pressing any key in auto mode switches to manual mode.)  Ctrl+C also quits.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import os
import sys
import time

from backend.schemas.models import Configuration, ConfigurationUpdate, WorkloadState
from simulation.simulator import GPUSimulator

try:
    import msvcrt          # Windows only
except ImportError:
    msvcrt = None

os.system("")              # enables ANSI colours in the Windows terminal
RED, GREEN, YELLOW, CYAN, DIM, RESET = "\033[91m", "\033[92m", "\033[93m", "\033[96m", "\033[2m", "\033[0m"

HEAD = (f"{'time':<8} {'id':>4} {'state':<9} {'req/min':>7} {'util%':>6} {'power W':>8} "
        f"{'temp C':>7} {'mem GB':>7} {'lat ms':>7} {'thr/min':>8}  SLA   utilization")


def fmt(s):
    ok = s.sla_status == "PASS"
    lat_c = GREEN if s.latency_ok else RED
    sla = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    bar = "#" * int(s.gpu_utilization / 5)
    return (f"{s.timestamp:%H:%M:%S} {s.sample_id:>4} {s.workload_state.value:<9} {s.request_rate:>7} "
            f"{s.gpu_utilization:>6.1f} {s.gpu_power:>8.1f} {s.temperature:>7.1f} {s.memory:>7.2f} "
            f"{lat_c}{s.latency:>7.1f}{RESET} {s.throughput:>8}  {sla}  {bar}")


def say(msg, colour=CYAN):
    print(f"{colour}>>> {msg}{RESET}")


def gpupilot_choose(sim):
    """Tiny stand-in 'brain': try candidate configs, keep the cheapest feasible one."""
    rate = sim.latest.request_rate
    best = None
    for prec in ("FP16", "INT8"):
        for batch in (16, 32, 64):
            for rep in (1, 2):
                c = Configuration(precision=prec, batch_size=batch, replicas=rep)
                e = sim.estimate_steady_state(c, rate)
                if e.feasible and (best is None or e.total_power < best[0].total_power):
                    best = (e, c)
    return best


def read_key():
    if msvcrt and msvcrt.kbhit():
        return msvcrt.getwch().lower()
    return None


def comparison_table(sim):
    before, after, delta, status = sim.comparison()
    if not after:
        return
    print(f"\n{YELLOW}BEFORE vs AFTER the configuration change (avg of 5 samples each){RESET}")
    print(f"    {'metric':<16}{'before':>10}{'after':>10}{'change':>10}")
    for k in ["gpu_utilization", "gpu_power", "temperature", "memory", "latency", "throughput"]:
        print(f"    {k:<16}{before[k]:>10.1f}{after[k]:>10.1f}{delta[k]:>+10.1f}")
    print(f"    SLA now: {status}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    a = ap.parse_args()

    sim = GPUSimulator()
    auto = not a.manual
    phase = "normal"            # auto script: normal -> spike -> fixing -> hold -> recovery -> done
    phase_ticks = fail_streak = rows = 0
    print(f"{YELLOW}=== GPUPilot LIVE  -  SIMULATED GPU TELEMETRY  -  1 reading every {a.interval:g}s ==={RESET}")
    print(f"{DIM}keys: s=spike  o=optimize  d=default config  r=recovery  q=quit   (Ctrl+C also quits){RESET}")
    if auto:
        say("Auto demo: normal -> workload spike -> GPUPilot detects SLA FAIL and fixes it")
    next_t = time.monotonic()

    try:
        while True:
            # ---- keyboard
            key = read_key()
            if key:
                if auto:
                    auto = False
                    say("Manual mode (auto demo stopped)", YELLOW)
                if key == "q":
                    break
                elif key == "s":
                    sim.trigger_spike(); say("WORKLOAD SPIKE triggered", RED)
                elif key == "o":
                    sim.apply_configuration(ConfigurationUpdate(precision="INT8", batch_size=32))
                    say("Config applied: INT8, batch 32", GREEN)
                elif key == "d":
                    sim.apply_configuration(ConfigurationUpdate(**{"precision": "FP16", "batch_size": 16, "replicas": 1, "gpu_allocation": 1, "concurrency": 32}))
                    say("Config reset to default (FP16, batch 16)", YELLOW)
                elif key == "r":
                    sim.trigger_recovery(); say("Recovery started: load decreasing", GREEN)

            # ---- one new reading
            s = sim.tick()
            if rows % 20 == 0:
                print(HEAD)
            print(fmt(s), flush=True)
            rows += 1
            phase_ticks += 1
            fail_streak = fail_streak + 1 if s.sla_status == "FAIL" else 0

            # ---- hands-free script
            if auto:
                if phase == "normal" and phase_ticks >= 10:
                    sim.trigger_spike(); say("WORKLOAD SPIKE triggered (judge clicked the button)", RED)
                    phase, phase_ticks = "spike", 0
                elif phase == "spike" and fail_streak >= 6:
                    say("GPUPilot: SLA violated for 6 readings in a row -> evaluating candidate configs...", CYAN)
                    best = gpupilot_choose(sim)
                    if best:
                        e, c = best
                        sim.apply_configuration(ConfigurationUpdate(**c.model_dump()))
                        say(f"GPUPilot applies: {c.precision}, batch {c.batch_size}, replicas {c.replicas} "
                            f"(predicted {e.latency:.0f} ms, {e.total_power:.0f} W total)", GREEN)
                    phase, phase_ticks = "hold", 0
                elif phase == "hold" and phase_ticks >= 20:
                    comparison_table(sim)
                    sim.trigger_recovery(); say("Spike over: load recovering", GREEN)
                    phase, phase_ticks = "recovery", 0
                elif phase == "recovery" and sim.state == WorkloadState.NORMAL and phase_ticks >= 8:
                    say("Demo complete. Run with --manual to play with it yourself.", YELLOW)
                    break

            next_t += a.interval
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    print(f"\n{DIM}Stopped. All values were SIMULATED GPU TELEMETRY.{RESET}")


if __name__ == "__main__":
    main()
