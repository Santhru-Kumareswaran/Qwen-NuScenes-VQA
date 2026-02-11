
import csv
from collections import Counter

CSV_PATH = "nuscenes_goal_tokens.csv"

def main():
    goals = []
    try:
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                goals.append(row['goal_name'])
    except FileNotFoundError:
        print(f"File {CSV_PATH} not found.")
        return

    total = len(goals)
    if total == 0:
        print("No samples found.")
        return

    counts = Counter(goals)
    
    print(f"{'Goal Token':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 46)
    
    # Sort by count descending
    for goal, count in counts.most_common():
        percentage = (count / total) * 100
        print(f"{goal:<20} | {count:<10} | {percentage:.2f}%")
        
    print("-" * 46)
    print(f"{'Total':<20} | {total:<10} | 100.00%")

if __name__ == "__main__":
    main()
