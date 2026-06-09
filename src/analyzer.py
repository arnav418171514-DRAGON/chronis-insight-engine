import pandas as pd
import numpy as np

class BehavioralInsightEngine:
    def __init__(self, data_path: str):
        """Initializes the engine and loads the behavioral data."""
        # Load the CSV and ensure dates are properly formatted for time-series math
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(by=['user_id', 'date'])

    def check_evidence_sufficiency(self, user_id: str, required_days: int = 7) -> dict:
        """
        Component 4: Evidence Sufficiency Gate.
        Evaluates if the user's data is robust enough to generate reliable insights.
        """
        user_data = self.df[self.df['user_id'] == user_id]
        
        # Gate 1: Volume Check (Do we have enough historical data?)
        if len(user_data) < required_days:
            return {
                "sufficient": False, 
                "reason": f"Insufficient volume: Only {len(user_data)} days logged. Minimum {required_days} required."
            }
            
        # Gate 2: Density & Completeness Check (Are there missing values?)
        core_metrics = ['steps', 'sleep_hours', 'screen_time_hours', 'deep_work_hours', 'exercise_minutes']
        missing_count = user_data[core_metrics].isnull().sum().sum()
        
        if missing_count > 0:
            return {
                "sufficient": False,
                "reason": f"Insufficient density: Data contains {missing_count} missing values. Abstaining from analysis."
            }
            
        return {"sufficient": True, "reason": "Data volume and density meet requirements."}

    def get_user_data(self, user_id: str):
        """Fetches user data ONLY if it passes the evidence sufficiency gate."""
        gate_check = self.check_evidence_sufficiency(user_id)
        
        if not gate_check["sufficient"]:
            print(f"ABSTAINING for {user_id}: {gate_check['reason']}")
            return None
            
        return self.df[self.df['user_id'] == user_id]