
import argparse
import pandas as pd
import json
from pathlib import Path

def load_ground_truth(csv_path):
    """
    Load ground truth violations.
    Expected CSV columns: video_name, frame_id, event_type, vehicle_type
    """
    df = pd.read_csv(csv_path)
    return df

def load_predictions(json_path):
    """
    Load system predictions from violations.json
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return pd.DataFrame()

def evaluate(gt_path, pred_path, time_tolerance_frames=30):
    """
    Match predictions to ground truth with a temporal tolerance.
    """
    gt = load_ground_truth(gt_path)
    pred = load_predictions(pred_path)
    
    if gt.empty:
        print("No ground truth data found.")
        return
    if pred.empty:
        print("No predictions found.")
        print(f"Precision: 0.0")
        print(f"Recall: 0.0")
        return

    # Normalize columns
    # We assume 'frame_id' is the key sync point
    
    tp = 0
    fp = 0
    fn = 0
    
    # Iterate through each GT event and try to find a match in Pred
    matched_pred_indices = set()
    
    for idx, row in gt.iterrows():
        # Filter preds by same type and roughly same frame (within tolerance)
        target_frame = row['frame_id']
        target_type = row['event_type']
        
        candidates = pred[
            (pred['violation_type'] == target_type) & 
            (pred['frame_id'] >= target_frame - time_tolerance_frames) &
            (pred['frame_id'] <= target_frame + time_tolerance_frames)
        ]
        
        # Find first unused match
        match_found = False
        for p_idx, p_row in candidates.iterrows():
            if p_idx not in matched_pred_indices:
                matched_pred_indices.add(p_idx)
                match_found = True
                break
        
        if match_found:
            tp += 1
        else:
            fn += 1
            
    # Remaining predictions are False Positives
    fp = len(pred) - len(matched_pred_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print("EVALUATION RESULTS")
    print("-" * 30)
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Violation Detection logic.")
    parser.add_argument("--gt", required=True, help="Path to Ground Truth CSV")
    parser.add_argument("--pred", required=True, help="Path to violations.json Prediction file")
    args = parser.parse_args()
    
    evaluate(args.gt, args.pred)
