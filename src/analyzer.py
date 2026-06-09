import pandas as pd
import numpy as np
from src.baselines import CLINICAL_BASELINES  # Importing our medical thresholds
from scipy.stats import linregress

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
    
    def discover_patterns(self, user_id: str, metric: str, window: int = 14, p_value_threshold: float = 0.05):
        """
        Component 1: Pattern Discovery.
        Uses Ordinary Least Squares (OLS) linear regression to identify 
        statistically significant long-term trends in behavior.
        """
        user_data = self.get_user_data(user_id)
        if user_data is None:
            return None  # Abstention gate triggered
            
        # We only want to look at the most recent 'window' of days
        recent_data = user_data.tail(window).copy()
        
        # We need at least 7 days to calculate a meaningful trend
        if len(recent_data) < 7:
             return None 
             
        # Create a numerical X-axis (days 0, 1, 2, 3...) for the regression
        x = np.arange(len(recent_data))
        y = recent_data[metric].values
        
        # Calculate the linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # If the p-value is > 0.05, the trend is likely just random noise. We ignore it.
        if p_value > p_value_threshold:
            return None
            
        # Determine the direction of the trend
        direction = "increasing" if slope > 0 else "decreasing"
        
        # Calculate absolute change from the start of the window to the end
        start_val = intercept # The modeled starting point
        end_val = intercept + (slope * len(recent_data))
        
        # Calculate our Data Completeness for the Confidence Score (Component 3)
        expected_days = window
        actual_days = len(recent_data.dropna(subset=[metric]))
        completeness_ratio = actual_days / expected_days
        
        # Base confidence on statistical significance (1 - p_value), penalized by missing data
        confidence = (1.0 - p_value) * completeness_ratio
        
        pattern = {
            "metric": metric,
            "trend": direction,
            "slope": slope,
            "p_value": p_value,
            "confidence": round(confidence, 2),
            "evidence": f"Over the last {window} days, {metric} showed a statistically significant {direction} trend (p={p_value:.3f}), shifting from ~{start_val:.1f} to ~{end_val:.1f}."
        }
        
        return pattern
    