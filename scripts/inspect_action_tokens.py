
import csv

CSV_PATH = "/home/santhru/FYP38_First Experiment/NuScenesVQA-/QWEN_VL_AD/output/nuscenes_action_tokens.csv"

def main():
    action_tokens = set()
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                at = row.get('action_token', '').strip()
                if at:
                    action_tokens.add(at)
        
        print("Unique Action Tokens found:")
        for at in sorted(list(action_tokens)):
            print(f"- {at}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
