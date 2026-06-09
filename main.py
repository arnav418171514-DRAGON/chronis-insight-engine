from src.analyzer import BehavioralInsightEngine
import json

def run_test():
    print("Initializing Chronis Behavioral Insight Engine...")
    engine = BehavioralInsightEngine("data/Chronis_TaskA_Synthetic_Behavioral_Data_v2-2.csv")
    
    user_to_test = "U3"
    metric_to_test = "steps"
    
    print(f"\n--- Testing Pattern Discovery for {user_to_test} on {metric_to_test} ---")
    
    pattern = engine.discover_patterns(user_id=user_to_test, metric=metric_to_test, window=14)
    
    if pattern is None:
        print("Test Result: No statistically significant long-term pattern found (or insufficient data).")
    else:
        print("Test Result: Found a significant behavioral pattern!")
        print(json.dumps(pattern, indent=4))

if __name__ == "__main__":
    run_test()