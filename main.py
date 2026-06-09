from src.analyzer import BehavioralInsightEngine
import json

def run_test():
    print("Initializing Chronis Behavioral Insight Engine...")
    
    # Point the engine to our data folder
    data_path = "data/Chronis_TaskA_Synthetic_Behavioral_Data_v2-2.csv"
    engine = BehavioralInsightEngine(data_path)
    
    # Let's test User U1 for sleep anomalies
    user_to_test = "U1"
    metric_to_test = "sleep_hours"
    
    print(f"\n--- Testing Anomaly Detection for {user_to_test} on {metric_to_test} ---")
    
    # Run the function we just built!
    anomalies = engine.detect_anomalies(user_id=user_to_test, metric=metric_to_test, window=7)
    
    # Print the results beautifully
    if anomalies is None:
        print("Test Result: Engine abstained from analysis due to insufficient data.")
    elif len(anomalies) == 0:
        print("Test Result: No statistically significant anomalies detected.")
    else:
        print(f"Test Result: Found {len(anomalies)} anomalies!")
        # Print them out formatted nicely
        print(json.dumps(anomalies, indent=4))

if __name__ == "__main__":
    run_test()