"""Integration test: parallel requests to multiple Ollama hosts.

Uses a small fast model (llama3.2:latest) so the test completes in
seconds, not minutes. Tests that round-robin distributes across hosts
and that failover works when a host is unreachable.

Requires: at least one Ollama host running with llama3.2:latest.
Configure hosts in ~/.config/beqdesigner/settings.json under
"ollama_hosts", or set OLLAMA_HOSTS env var.

Skipped if no Ollama host is reachable.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from model.auto_beq_advisor import (
    CurveFeatures,
    MediaMetadata,
    OllamaAdvisor,
    _load_ollama_hosts,
)

log = logging.getLogger("test_ollama_multihost")

# Small fast model for integration tests — responds in <5s.
_FAST_MODEL = "llama3.2:latest"


def _host_has_model(host: str, model: str) -> bool:
    try:
        url = f"{host.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return any(m["name"] == model for m in data.get("models", []))
    except Exception:
        return False


def _reachable_hosts() -> list[str]:
    """Return hosts from config that are reachable and have the fast model."""
    return [h for h in _load_ollama_hosts() if _host_has_model(h, _FAST_MODEL)]


def _dummy_features() -> CurveFeatures:
    return CurveFeatures(
        shoulder_peak_db=5.0, shoulder_peak_hz=25.0,
        level_at_5hz_db=-10.0, level_at_10hz_db=-5.0, level_at_20hz_db=4.0,
        rolloff_depth_db=15.0, rolloff_slope_db_per_oct=5.0,
        dynamic_range_db=18.0,
        curve_sample_points=tuple((float(hz), 0.0) for hz in (5, 10, 20, 40, 80)),
    )


@pytest.fixture(scope="module")
def live_hosts():
    hosts = _reachable_hosts()
    if not hosts:
        pytest.skip(
            f"no Ollama host reachable with {_FAST_MODEL}. "
            f"Configured hosts: {_load_ollama_hosts()}"
        )
    return hosts


class TestSingleHost:
    """Basic: one host, one request, valid JSON back."""

    def test_single_call_returns_json(self, live_hosts):
        advisor = OllamaAdvisor(
            hosts=[live_hosts[0]], model=_FAST_MODEL, timeout_s=30,
        )
        meta = MediaMetadata(title="Test Film", year=2024)
        advice = advisor.advise(meta, _dummy_features())
        assert advice.source.startswith("ollama:")
        assert advice.max_gain_db >= 0
        log.info("single host: %s -> max_gain=%.1f knee=%s",
                 live_hosts[0], advice.max_gain_db, advice.knee_hz)


class TestMultiHost:
    """Requires 2+ reachable hosts."""

    @pytest.fixture(autouse=True)
    def _require_two_hosts(self, live_hosts):
        if len(live_hosts) < 2:
            pytest.skip(f"need 2+ hosts, have {len(live_hosts)}: {live_hosts}")

    def test_round_robin_distributes(self, live_hosts):
        """Fire N requests — each host should get at least one."""
        OllamaAdvisor.reset_host_stats()
        advisor = OllamaAdvisor(
            hosts=live_hosts, model=_FAST_MODEL, timeout_s=30,
        )
        meta = MediaMetadata(title="Round Robin Test", year=2024)
        features = _dummy_features()

        n_requests = len(live_hosts) * 2
        for _ in range(n_requests):
            advisor.advise(meta, features)

        OllamaAdvisor.print_host_stats()
        # Each host should have been called at least once.
        from model.auto_beq_advisor import _OLLAMA_HOST_STATS
        for host in live_hosts:
            assert host in _OLLAMA_HOST_STATS, f"host {host} never called"
            assert _OLLAMA_HOST_STATS[host]["calls"] >= 1

    def test_parallel_requests(self, live_hosts):
        """Fire requests in parallel — one thread per host."""
        OllamaAdvisor.reset_host_stats()

        def _call_advisor(title: str):
            advisor = OllamaAdvisor(
                hosts=live_hosts, model=_FAST_MODEL, timeout_s=30,
            )
            return advisor.advise(
                MediaMetadata(title=title, year=2024),
                _dummy_features(),
            )

        titles = [f"Parallel Test {i}" for i in range(len(live_hosts) * 2)]
        results = []
        with ThreadPoolExecutor(max_workers=len(live_hosts)) as pool:
            futures = {pool.submit(_call_advisor, t): t for t in titles}
            for future in as_completed(futures):
                advice = future.result()
                results.append(advice)
                log.info("parallel: %s -> max_gain=%.1f source=%s",
                         futures[future], advice.max_gain_db, advice.source)

        assert len(results) == len(titles)
        OllamaAdvisor.print_host_stats()


class TestFailover:
    """One real host + one fake host — requests should still succeed."""

    def test_failover_to_working_host(self, live_hosts):
        """Include a dead host — advisor should fail over to the live one."""
        OllamaAdvisor.reset_host_stats()
        hosts_with_dead = ["http://192.168.1.254:11434"] + live_hosts
        advisor = OllamaAdvisor(
            hosts=hosts_with_dead, model=_FAST_MODEL, timeout_s=5,
        )
        meta = MediaMetadata(title="Failover Test", year=2024)
        advice = advisor.advise(meta, _dummy_features())
        assert advice.source.startswith("ollama:")
        OllamaAdvisor.print_host_stats()
        log.info("failover: dead host skipped, got max_gain=%.1f", advice.max_gain_db)
