"""Unit tests for KoeAI LLVC candidate evaluation."""

from hth.adapters.llvc.evaluator import audit_llvc_candidate


def test_llvc_audit_structure():
    result = audit_llvc_candidate()
    assert result.repo_url == "https://github.com/KoeAI/LLVC"
    assert result.upstream_license == "MIT"
    assert result.is_commercial_cleared is True
    assert result.checkpoint_size_mb < 50.0  # ~37.66 MB
    assert result.female_target_available is True
    assert result.male_target_available is False
    assert result.realtime_10ms_budget_met is False  # Honest CPU assessment


def test_llvc_disk_reserve_verification():
    result = audit_llvc_candidate()
    # On this machine, free disk space is ~10.5 GB, so 25 GB reserve is not met
    assert result.disk_reserve_satisfied is False
    assert result.free_disk_gb < 25.0
