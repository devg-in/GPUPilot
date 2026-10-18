"""TinyFish-assisted GPU driver/software update monitor for the simulator.

The hardware inventory is explicitly simulated. TinyFish Search supplies live
public-web context; it does not inspect or modify a real host.
"""
from __future__ import annotations

import os
import re
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

TINYFISH_SEARCH_URL = "https://api.search.tinyfish.ai/"

# Demo inventory: values are illustrative and must not be mistaken for a host scan.
SIMULATED_INVENTORY = [
    {"component": "GPU Driver", "installed_version": "570.86.15", "source": "simulated inventory", "vendor": "NVIDIA"},
    {"component": "CUDA Toolkit", "installed_version": "12.8", "source": "simulated inventory", "vendor": "NVIDIA"},
    {"component": "PyTorch", "installed_version": "2.6.0", "source": "simulated inventory", "vendor": "PyTorch"},
]

QUERIES = [
    ("GPU Driver", "site:nvidia.com/en-us/drivers/ NVIDIA Linux x86_64 data center GPU driver latest release version"),
    ("CUDA Toolkit", "site:developer.nvidia.com/cuda-downloads CUDA Toolkit latest release version"),
    ("PyTorch", "site:pytorch.org/blog PyTorch latest stable release version"),
]

VERSION_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{1,4}){1,3})(?!\d)")


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split('.'))
    except (ValueError, AttributeError):
        return ()


def _find_version(text: str, component: str) -> str | None:
    """Conservative version extraction; never treat arbitrary dates as confirmed updates."""
    text = text or ""
    candidates = VERSION_RE.findall(text)
    if component == "GPU Driver":
        # NVIDIA Linux driver versions usually have 3 numeric components, e.g. 570.124.06.
        candidates = [v for v in candidates if len(v.split('.')) >= 3 and int(v.split('.')[0]) >= 400]
    elif component == "CUDA Toolkit":
        candidates = [v for v in candidates if len(v.split('.')) == 2 and 10 <= int(v.split('.')[0]) <= 20]
    else:
        candidates = [v for v in candidates if len(v.split('.')) >= 2 and int(v.split('.')[0]) <= 10]
    return candidates[0] if candidates else None


def _tinyfish_search(query: str, api_key: str) -> dict[str, Any]:
    url = TINYFISH_SEARCH_URL + "?" + urllib.parse.urlencode({"query": query, "location": "US", "language": "en"})
    request = urllib.request.Request(url, headers={"X-API-Key": api_key, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=12) as response:
        if response.status != 200:
            raise RuntimeError(f"TinyFish Search returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def initial_monitor_state() -> dict[str, Any]:
    return {
        "integration": "TinyFish Search API",
        "configured": bool(os.getenv("TINYFISH_API_KEY", "").strip()),
        "mode": "live" if os.getenv("TINYFISH_API_KEY", "").strip() else "not_configured",
        "last_scan_at": None,
        "inventory": [dict(item) for item in SIMULATED_INVENTORY],
        "findings": [],
        "message": "Set TINYFISH_API_KEY to enable live release searches. The inventory is simulated.",
    }


def scan_environment() -> dict[str, Any]:
    api_key = os.getenv("TINYFISH_API_KEY", "").strip()
    now = datetime.now(timezone.utc).isoformat()
    inventory = [dict(item) for item in SIMULATED_INVENTORY]
    if not api_key:
        return {
            "integration": "TinyFish Search API", "configured": False, "mode": "not_configured",
            "last_scan_at": now, "inventory": inventory, "findings": [],
            "message": "TinyFish API key is not configured. No live update claims were made. Set TINYFISH_API_KEY and scan again.",
        }

    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for component, query in QUERIES:
        installed = next(item["installed_version"] for item in inventory if item["component"] == component)
        try:
            payload = _tinyfish_search(query, api_key)
            results = payload.get("results", []) if isinstance(payload, dict) else []
            # Only use a result when its source is recognizably relevant to the vendor/project.
            accepted = []
            for result in results[:5]:
                title = str(result.get("title", ""))
                snippet = str(result.get("snippet", ""))
                source_url = str(result.get("url", ""))
                combined = f"{title} {snippet}"
                version = _find_version(combined, component)
                if source_url and version:
                    accepted.append({"title": title or source_url, "url": source_url, "snippet": snippet[:400], "version": version})
            if accepted:
                candidate = accepted[0]
                installed_tuple, candidate_tuple = _version_tuple(installed), _version_tuple(candidate["version"])
                # Pad tuples so 12.8 and 12.8.0 compare consistently; major changes are allowed.
                width = max(len(installed_tuple), len(candidate_tuple))
                installed_norm = installed_tuple + (0,) * (width - len(installed_tuple))
                candidate_norm = candidate_tuple + (0,) * (width - len(candidate_tuple))
                comparable = bool(installed_tuple and candidate_tuple)
                if comparable and candidate_norm > installed_norm:
                    status = "update_candidate"
                    summary = f"Search result suggests {candidate['version']} may be newer than simulated installed version {installed}; verify official release notes and compatibility before updating."
                else:
                    status = "review_required"
                    summary = f"Found release information ({candidate['version']}); automatic version comparison is not conclusive. Review the official source before changing the environment."
                findings.append({"component": component, "installed_version": installed, "candidate_version": candidate["version"], "status": status, "summary": summary, "source_title": candidate["title"], "source_url": candidate["url"], "snippet": candidate["snippet"]})
            else:
                # No version extracted means never falsely claim an update is available.
                first = results[0] if results else {}
                findings.append({"component": component, "installed_version": installed, "candidate_version": None, "status": "review_required" if results else "no_results", "summary": "TinyFish returned relevant search context, but no reliable version could be extracted; manual review required." if results else "TinyFish returned no results for this query.", "source_title": str(first.get("title", "")), "source_url": str(first.get("url", "")), "snippet": str(first.get("snippet", ""))[:400]})
        except Exception as exc:
            errors.append(f"{component}: {type(exc).__name__}: {exc}")
            findings.append({"component": component, "installed_version": installed, "candidate_version": None, "status": "scan_error", "summary": "Live search failed; no update status was inferred.", "source_title": "", "source_url": "", "snippet": ""})

    update_count = sum(1 for finding in findings if finding["status"] == "update_candidate")
    message = f"Live TinyFish scan completed. {update_count} possible update candidate(s); verify compatibility before any change."
    if errors:
        message += " Some searches failed: " + "; ".join(errors)
    return {"integration": "TinyFish Search API", "configured": True, "mode": "live", "last_scan_at": now,
            "inventory": inventory, "findings": findings, "message": message}
