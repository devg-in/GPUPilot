"""Lightweight workload/thermal prediction for the simulator-first MVP."""
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False


@dataclass
class PredictionResult:
    mode: str
    trained_samples: int
    next_request_rate: float
    predicted_power: float
    predicted_temperature: float
    predicted_latency: float
    workload_intensity: str
    confidence: float
    features: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_source": "SIMULATED GPU TELEMETRY",
            "mode": self.mode,
            "trained_samples": self.trained_samples,
            "next_request_rate": round(self.next_request_rate, 1),
            "predicted_power": round(self.predicted_power, 1),
            "predicted_temperature": round(self.predicted_temperature, 1),
            "predicted_latency": round(self.predicted_latency, 1),
            "workload_intensity": self.workload_intensity,
            "confidence": round(self.confidence, 2),
            "features": {k: round(float(v), 2) for k, v in self.features.items()},
        }


class GPUPilotPredictor:
    """Predict the next operating point from recent simulated telemetry.

    With enough history and scikit-learn installed, RandomForest models are
    retrained on the current history. Otherwise a deterministic trend fallback
    is used, so the hackathon demo never depends on ML availability.
    """

    def __init__(self, min_samples: int = 25):
        self.min_samples = min_samples

    @staticmethod
    def _features(sample) -> List[float]:
        return [sample.request_rate, sample.gpu_utilization, sample.temperature,
                sample.memory, sample.latency, sample.throughput]

    @staticmethod
    def _trend(values: List[float], default: float) -> float:
        if len(values) < 2:
            return default
        window = values[-8:]
        delta = (window[-1] - window[0]) / max(len(window) - 1, 1)
        return window[-1] + delta

    def predict(self, samples) -> PredictionResult:
        samples = list(samples)
        if not samples:
            return PredictionResult("fallback", 0, 0, 0, 0, 0, "UNKNOWN", 0.0, {})

        current = samples[-1]
        feature = np.array([self._features(current)], dtype=float)
        mode = "fallback"
        confidence = 0.45

        req_values = [float(s.request_rate) for s in samples]
        power_values = [float(s.gpu_power) for s in samples]
        temp_values = [float(s.temperature) for s in samples]
        lat_values = [float(s.latency) for s in samples]

        next_req = self._trend(req_values, current.request_rate)
        next_power = self._trend(power_values, current.gpu_power)
        next_temp = self._trend(temp_values, current.temperature)
        next_lat = self._trend(lat_values, current.latency)

        if SKLEARN_AVAILABLE and len(samples) >= self.min_samples:
            try:
                # Predict the next sample from the current telemetry state.
                # Training target is the following sample for each historical row.
                X = np.array([self._features(s) for s in samples[:-1]], dtype=float)
                y_power = np.array([s.gpu_power for s in samples[1:]], dtype=float)
                y_temp = np.array([s.temperature for s in samples[1:]], dtype=float)
                y_req = np.array([s.request_rate for s in samples[1:]], dtype=float)
                y_lat = np.array([s.latency for s in samples[1:]], dtype=float)
                if len(X) >= self.min_samples:
                    kwargs = {"n_estimators": 80, "random_state": 42, "min_samples_leaf": 2}
                    m_power = RandomForestRegressor(**kwargs).fit(X, y_power)
                    m_temp = RandomForestRegressor(**kwargs).fit(X, y_temp)
                    m_req = RandomForestRegressor(**kwargs).fit(X, y_req)
                    m_lat = RandomForestRegressor(**kwargs).fit(X, y_lat)
                    next_power = float(m_power.predict(feature)[0])
                    next_temp = float(m_temp.predict(feature)[0])
                    next_req = float(m_req.predict(feature)[0])
                    next_lat = float(m_lat.predict(feature)[0])
                    mode = "random_forest"
                    confidence = min(0.95, 0.55 + min(len(samples), 200) / 1000)
            except Exception:
                mode = "fallback"
                confidence = 0.45

        if next_req < current.request_rate * 0.97:
            intensity = "DECREASING"
        elif next_req > current.request_rate * 1.03:
            intensity = "INCREASING"
        else:
            intensity = "STABLE"

        return PredictionResult(
            mode=mode,
            trained_samples=max(0, len(samples) - 1),
            next_request_rate=max(0.0, next_req),
            predicted_power=max(0.0, next_power),
            predicted_temperature=max(0.0, next_temp),
            predicted_latency=max(0.0, next_lat),
            workload_intensity=intensity,
            confidence=confidence,
            features={
                "request_rate": current.request_rate,
                "gpu_utilization": current.gpu_utilization,
                "gpu_power": current.gpu_power,
                "temperature": current.temperature,
                "latency": current.latency,
                "throughput": current.throughput,
            },
        )
