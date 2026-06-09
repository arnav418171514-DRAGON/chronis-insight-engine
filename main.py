from src.analyzer import BehavioralInsightEngine

def run_test():
    print("Initializing Chronis Behavioral Insight Engine...\n")
    engine = BehavioralInsightEngine("data/Chronis_TaskA_Synthetic_Behavioral_Data_v2-2.csv")
    
    users = ["U1", "U2", "U3", "U4", "U5"]
    
    for user in users:
        print(f"--- Behavioral Report: {user} ---")
        insights = engine.generate_insights(user)
        
        for item in insights:
            print(f"Insight: {item.get('insight', 'No insight generated.')}")
            
            # Only print confidence if it's a trend or anomaly (abstentions don't have confidence)
            if "confidence" in item:
                print(f"Confidence: {item.get('confidence')}")
                
            # Handle both 'evidence' and 'reason' keys depending on the insight type
            if "evidence" in item:
                print(f"Evidence: {item.get('evidence')}")
            elif "reason" in item:
                print(f"Evidence: {item.get('reason')}")
                
            print("-" * 20) # A small divider between multiple insights for one user
            
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_test()