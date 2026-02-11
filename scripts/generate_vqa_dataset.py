
import csv
import json
import os

# ================= CONFIG =================
INPUT_CSV = "nuscenes_goal_tokens_refined.csv"
OUTPUT_JSON = "nuscenes_action_prediction_vqa.json"
# ==========================================

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    vqa_data = []
    
    print(f"Reading {INPUT_CSV}...")
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_token = row['sample_token']
            goal_name = row['goal_name']
            action_token = row['action_token']
            
            # Skip if goal or action is missing/unknown
            if not goal_name or not action_token or goal_name == "UNKNOWN":
                continue

            maneuver_type = row['maneuver_type']

            # Detailed Question (Minimal)
            question_dict = {
                "goal": goal_name
            }
            question_text = json.dumps(question_dict)

            # Detailed Answer (Action + Maneuver)
            answer_dict = {
                "action": action_token,
                "maneuver": maneuver_type
            }
            answer_text = json.dumps(answer_dict)

            entry = {
                "split": "train",
                "sample_token": sample_token,
                "question": question_text,
                "answer": answer_text
            }
            vqa_data.append(entry)

    print(f"Generated {len(vqa_data)} VQA entries.")
    
    print(f"Writing to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(vqa_data, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()
