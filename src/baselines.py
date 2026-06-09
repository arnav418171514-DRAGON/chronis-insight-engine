"""
Clinical & Psychological Baselines for Behavioral Metrics.
Sourced from established health organizations (WHO, CDC, NIH) to ensure 
anomaly detection is grounded in medical reality, not arbitrary developer thresholds.
"""

CLINICAL_BASELINES = {
    "sleep_hours": {
        "min_threshold": 7.0,  # CDC recommends 7-9 hours. <7 is sleep deprivation.
        "unit": "hours"
    },
    "steps": {
        "min_threshold": 8000,  # NIH associates 8,000+ steps with significantly lower mortality.
        "unit": "steps"
    },
    "exercise_minutes": {
        "min_threshold": 21,  # WHO recommends 150 mins/week (avg 21 mins/day).
        "unit": "minutes"
    },
    "deep_work_hours": {
        "max_threshold": 4.0,  # Cognitive threshold before burnout (based on deliberate practice research).
        "unit": "hours"
    },
    "screen_time_hours": {
        "max_threshold": 6.0,  # High non-work screen time correlates with circadian disruption.
        "unit": "hours"
    }
}