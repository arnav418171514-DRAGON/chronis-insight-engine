import pandas as pd
import numpy as np
from src.baselines import CLINICAL_BASELINES  # Importing our medical thresholds

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

    def detect_anomalies(self, user_id: str, metric: str, window: int = 7, z_threshold: float = 2.0):
        """
        Component 2: Anomaly Detection.
        Uses a rolling Z-score to find statistically significant deviations,
        then cross-references with clinical baselines.
        """
        user_data = self.get_user_data(user_id)
        if user_data is None:
            return None  # Abstention gate triggered

        # Create a copy to do our math on
        df_metric = user_data[['date', metric]].copy()
        
        # Calculate rolling mean and standard deviation for the past 'window' days
        df_metric['rolling_mean'] = df_metric[metric].rolling(window=window, min_periods=window).mean().shift(1)
        df_metric['rolling_std'] = df_metric[metric].rolling(window=window, min_periods=window).std().shift(1)
        
        # Calculate Z-score. We use np.where to prevent division by zero errors if standard deviation is 0.
        df_metric['z_score'] = np.where(
            df_metric['rolling_std'] > 0,
            (df_metric[metric] - df_metric['rolling_mean']) / df_metric['rolling_std'],
            0
        )

        anomalies = []
        
        # Iterate through the data to find days where the Z-score exceeds our threshold
        for index, row in df_metric.dropna().iterrows():
            if abs(row['z_score']) > z_threshold:
                val = row[metric]
                date_str = row['date'].strftime('%Y-%m-%d')
                
                # Check against our clinical baselines
                baseline_info = CLINICAL_BASELINES.get(metric, {})
                min_thresh = baseline_info.get('min_threshold')
                max_thresh = baseline_info.get('max_threshold')
                
                reason = f"Statistically unusual event (Z={row['z_score']:.2f}). "
                
                # Add clinical context to the explanation
                if min_thresh and val < min_thresh and row['z_score'] < 0:
                    reason += f"Value ({val}) dropped dangerously below the clinical minimum of {min_thresh}."
                elif max_thresh and val > max_thresh and row['z_score'] > 0:
                    reason += f"Value ({val}) exceeded the clinical maximum of {max_thresh}."
                else:
                    reason += f"Value was {val}, compared to the user's recent rolling average of {row['rolling_mean']:.2f}."

                anomalies.append({
                    "date": date_str,
                    "metric": metric,
                    "value": val,
                    "reason": reason
                })
                
        return anomalies