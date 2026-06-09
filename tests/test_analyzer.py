import pytest
from src.analyzer import BehavioralInsightEngine

# A pytest fixture runs once and provides the engine to all our test functions
@pytest.fixture
def engine():
    return BehavioralInsightEngine("data/Chronis_TaskA_Synthetic_Behavioral_Data_v2-2.csv")

def test_evidence_sufficiency_abstention(engine):
    """Component 4: Test that the system abstains when a user has no data."""
    # A user that doesn't exist will have 0 rows of data
    gate = engine.check_evidence_sufficiency("NON_EXISTENT_USER")
    assert gate["sufficient"] is False
    assert "Insufficient volume" in gate["reason"]

def test_evidence_sufficiency_pass(engine):
    """Component 4: Test that a valid user passes the gate."""
    gate = engine.check_evidence_sufficiency("U1")
    assert gate["sufficient"] is True

def test_anomaly_detection_logic(engine):
    """Component 2: Test if the math and clinical baselines trigger correctly."""
    # We know from our manual run that U2 had an exercise anomaly on 2026-01-29
    anomalies = engine.detect_anomalies("U2", "exercise_minutes", window=7)
    
    assert anomalies is not None
    assert len(anomalies) > 0
    
    # Check if the specific date was flagged
    flagged_dates = [a["date"] for a in anomalies]
    assert "2026-01-29" in flagged_dates
    
    # Verify the clinical baseline explanation is generated correctly
    specific_anomaly = next(a for a in anomalies if a["date"] == "2026-01-29")
    assert "dangerously below the clinical minimum of 21" in specific_anomaly["reason"]

def test_pattern_discovery_trend(engine):
    """Component 1: Test if linear regression properly identifies a valid trend."""
    # We know U4 has an increasing trend in steps
    pattern = engine.discover_patterns("U4", "steps", window=14)
    
    assert pattern is not None
    assert pattern["trend"] == "increasing"
    assert pattern["p_value"] < 0.05  # Must be statistically significant
    assert pattern["confidence"] > 0.8  # Must have high confidence

def test_pattern_discovery_no_trend(engine):
    """Component 1: Test if linear regression correctly ignores noisy/stable data."""
    # U3 has stable steps with no significant long-term trend
    pattern = engine.discover_patterns("U3", "steps", window=14)
    assert pattern is None