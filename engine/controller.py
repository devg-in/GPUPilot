"""Closed-loop GPUPilot controller."""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.schemas.models import ConfigurationUpdate
from engine.sustainability import compare_impact


class GPUPilotController:
    def __init__(self, simulator, optimizer, predictor, interval_s: float = 2.0,
                 min_improvement_pct: float = 1.5, cooldown_s: float = 3.0):
        self.simulator = simulator
        self.optimizer = optimizer
        self.predictor = predictor
        self.interval_s = interval_s
        self.min_improvement_pct = min_improvement_pct
        self.cooldown_s = cooldown_s
        self.task: Optional[asyncio.Task] = None
        self.last_action_at: Optional[float] = None
        self.last_cycle: Optional[Dict[str, Any]] = None
        self.events: List[Dict[str, Any]] = []
        self.max_events = 100
        self.last_gpu_selection = []

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def _event(self, event: str, message: str, **details):
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "message": message,
            **details,
        }
        self.events.append(item)
        self.events = self.events[-self.max_events:]
        return item

    @staticmethod
    def _trigger_reason(telemetry, result) -> str:
        reasons = []
        if telemetry.sla_status == "FAIL":
            reasons.append("SLA degradation")
        if telemetry.gpu_power >= 275:
            reasons.append("high GPU power")
        if telemetry.temperature >= 78:
            reasons.append("thermal pressure")
        if telemetry.request_rate >= 950:
            reasons.append("workload pressure")
        if not reasons and result.recommended_configuration != result.current_configuration:
            reasons.append("sustainability inefficiency")
        return ", ".join(reasons) if reasons else "adaptive optimization"

    async def run_cycle(self) -> Dict[str, Any]:
        telemetry = self.simulator.latest
        prediction = self.predictor.predict(self.simulator.get_history(include_live=True, limit=300))
        selected_gpus = [g["id"] for g in self.simulator.get_cluster() if g["active"]]
        if selected_gpus != self.last_gpu_selection:
            self._event("gpu_allocation_changed", "Dynamic GPU scheduler updated the active GPU set.",
                        previous=self.last_gpu_selection, selected=selected_gpus,
                        workload_state=self.simulator.state.value, request_rate=telemetry.request_rate)
            self.last_gpu_selection = selected_gpus
        result = await asyncio.to_thread(self.optimizer.optimize)
        current = result.current_configuration
        recommended = result.recommended_configuration

        action = "NO_OP"
        applied = current
        impact = None
        reason = result.reason

        current_impact = result.current_estimate
        recommended_impact = result.recommended_estimate
        power_improvement = 0.0
        if current_impact.total_power:
            power_improvement = ((current_impact.total_power - recommended_impact.total_power)
                                 / current_impact.total_power) * 100.0

        now = asyncio.get_running_loop().time()
        cooldown_ok = self.last_action_at is None or (now - self.last_action_at) >= self.cooldown_s

        post_change_estimate = result.current_estimate
        required_gpus = self.optimizer.required_gpu_count(self.simulator.latest.request_rate)
        current_gpus = current.gpu_allocation * current.replicas
        recommended_gpus = recommended.gpu_allocation * recommended.replicas
        scale_up_required = recommended_gpus > current_gpus and recommended_gpus >= required_gpus
        scale_down = recommended_gpus < current_gpus and recommended_gpus >= required_gpus
        # Keep extra capacity while demand is still elevated, but do not freeze
        # the configuration for the entire recovery state. Once measured demand
        # falls below the pressure threshold and the recommended footprint still
        # meets the current GPU floor, allow the next short control cycle to scale
        # down. This avoids the long unchanged-config period seen in live demos.
        recovery_hold = (
            scale_down
            and self.simulator.state.value in {"spike", "peak", "recovery"}
            and (self.simulator.latest.request_rate >= 950
                 or self.simulator.latest.sla_status == "FAIL")
        )
        should_apply = (
            recommended != current
            and result.feasible_candidates > 0
            and cooldown_ok
            and not recovery_hold
            and (scale_up_required or (scale_down and power_improvement >= self.min_improvement_pct)
                 or (not scale_up_required and not scale_down and power_improvement >= self.min_improvement_pct))
        )

        if should_apply:
            previous = current.model_copy()
            update = ConfigurationUpdate(**recommended.model_dump())
            applied = self.simulator.apply_configuration(update)
            post_change_estimate = self.simulator.estimate_steady_state(applied, self.simulator.latest.request_rate)
            if not post_change_estimate.feasible:
                self.simulator.apply_configuration(ConfigurationUpdate(**previous.model_dump()))
                applied = previous
                action = "ROLLED_BACK"
                reason = "Post-change SLA verification failed; GPUPilot restored the previous safe configuration."
                self._event("rollback", reason, before=previous.model_dump(), attempted=recommended.model_dump())
            else:
                action = "APPLIED"
                self.last_action_at = now
                impact = compare_impact(result.current_estimate, post_change_estimate)
                self._event(
                    "configuration_applied",
                    "Applied a lower-impact feasible configuration after SLA verification.",
                    trigger=self._trigger_reason(telemetry, result),
                    before=previous.model_dump(),
                    after=applied.model_dump(),
                    power_before_w=round(result.current_estimate.total_power, 2),
                    power_after_w=round(post_change_estimate.total_power, 2),
                    latency_before_ms=round(result.current_estimate.latency, 2),
                    latency_after_ms=round(post_change_estimate.latency, 2),
                    throughput_before_req_min=round(result.current_estimate.throughput, 2),
                    throughput_after_req_min=round(post_change_estimate.throughput, 2),
                    sla_status=post_change_estimate.sla.sla_status,
                    power_improvement_pct=round(power_improvement, 2),
                )
        elif recommended != current and not cooldown_ok:
            action = "COOLDOWN"
            reason = "A safer configuration was found, but the controller is in cooldown to prevent thrashing."
            self._event("cooldown", reason)
        elif recommended != current and power_improvement < self.min_improvement_pct:
            action = "NO_OP"
            reason = f"Candidate improvement {power_improvement:.2f}% is below the {self.min_improvement_pct:.1f}% action threshold."
        else:
            self._event("no_change", "Current configuration is already a suitable feasible operating point.")

        self.last_cycle = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "telemetry": telemetry.model_dump(mode="json"),
            "prediction": prediction.as_dict(),
            "current_configuration": current.model_dump(),
            "recommended_configuration": recommended.model_dump(),
            "applied_configuration": applied.model_dump(),
            "current_estimate": result.current_estimate.model_dump(),
            "recommended_estimate": result.recommended_estimate.model_dump(),
            "post_change_estimate": post_change_estimate.model_dump(),
            "candidates_checked": result.candidates_checked,
            "feasible_candidates": result.feasible_candidates,
            "power_improvement_pct": round(power_improvement, 2),
            "reason": reason,
            "impact": impact,
            "selected_gpus": selected_gpus,
            "required_gpus": required_gpus,
            "current_gpu_count": current_gpus,
            "recommended_gpu_count": recommended_gpus,
        }
        return self.last_cycle

    async def _loop(self):
        while True:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._event("error", f"Controller cycle failed: {exc}")
            await asyncio.sleep(self.interval_s)

    def start(self, interval_s: Optional[float] = None):
        if interval_s is not None:
            self.interval_s = max(1.0, min(float(interval_s), 60.0))
        if not self.running:
            self.task = asyncio.create_task(self._loop())
            self._event("autopilot_started", f"Continuous autopilot started at {self.interval_s:.1f}s interval.")

    async def stop(self):
        if self.running:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        self._event("autopilot_stopped", "Continuous autopilot stopped.")

    def status(self):
        return {
            "running": self.running,
            "interval_s": self.interval_s,
            "min_improvement_pct": self.min_improvement_pct,
            "cooldown_s": self.cooldown_s,
            "last_cycle": self.last_cycle,
            "recent_events": self.events[-20:],
        }
