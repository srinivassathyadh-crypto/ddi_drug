"""
Prediction Logger Module
Shared logging utilities for automatic data collection
"""

import json
import os
from datetime import datetime
from typing import Optional

HISTORY_FILE = "prediction_history.json"
FEEDBACK_FILE = "user_feedback.json"

def log_prediction(
    drug1: str,
    drug2: str,
    model: str,
    predicted_type: int,
    confidence: float,
    prediction_time: float,
    ground_truth: Optional[int] = None,
    additional_data: Optional[dict] = None
):
    """
    Log a prediction to the history file

    Args:
        drug1: First drug ID
        drug2: Second drug ID
        model: Model name (DGNN, Random Forest, XGBoost)
        predicted_type: Predicted interaction type (1-86)
        confidence: Prediction confidence (0-100)
        prediction_time: Time taken for prediction in seconds
        ground_truth: Actual interaction type if known
        additional_data: Any extra data to store
    """
    try:
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "drug1": str(drug1),
            "drug2": str(drug2),
            "model": str(model),
            "predicted_type": int(predicted_type),
            "confidence": float(confidence),
            "prediction_time": float(prediction_time),
            "ground_truth": int(ground_truth) if ground_truth is not None else None,
            "correct": (int(predicted_type) == int(ground_truth)) if ground_truth is not None else None
        }

        # Add any additional data
        if additional_data:
            prediction_data.update(additional_data)

        # Load existing history
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # Append new prediction
        history.append(prediction_data)

        # Save back to file
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)

        return True
    except Exception as e:
        print(f"Error logging prediction: {e}")
        return False


def load_history():
    """Load all prediction history"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def get_stats():
    """Get quick statistics from history"""
    history = load_history()

    if not history:
        return {
            "total_predictions": 0,
            "models": [],
            "avg_confidence": 0,
            "accuracy": None
        }

    total = len(history)
    models = list(set([p['model'] for p in history]))
    avg_conf = sum([p['confidence'] for p in history]) / total

    with_gt = [p for p in history if p.get('ground_truth') is not None]
    accuracy = None
    if with_gt:
        correct = sum([1 for p in with_gt if p.get('correct')])
        accuracy = (correct / len(with_gt)) * 100

    return {
        "total_predictions": total,
        "models": models,
        "avg_confidence": avg_conf,
        "accuracy": accuracy,
        "with_ground_truth": len(with_gt)
    }


def clear_history():
    """Clear all prediction history (use with caution)"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return True
