from src.analyzer import BehavioralInsightEngine
import json

def run_test():
    print("Initializing Chronis Behavioral Insight Engine...\n")
    engine = BehavioralInsightEngine("data/Chronis_TaskA_Synthetic_Behavioral_Data_v2-2.csv")
    
    users = ["U1", "U2", "U3", "U4", "U5"]
    
    for user in users:
        print(f"--- Generating Insights for {user} ---")
        insights = engine.generate_insights(user)
        print(json.dumps(insights, indent=4))
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_test()