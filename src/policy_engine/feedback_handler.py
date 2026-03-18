import os
import json
from .policy_rules import get_rules, update_rules

FEEDBACK_LOG = 'data/feedback_log.json'

def record_feedback(prediction_prob, actual_label, explanation_quality):
    """
    Record user feedback and adapt policy thresholds.
    """
    feedback_entry = {
        "prediction_prob": prediction_prob,
        "actual_label": actual_label,
        "explanation_quality": explanation_quality
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, 'r') as f:
            logs = json.load(f)
            
    logs.append(feedback_entry)
    
    with open(FEEDBACK_LOG, 'w') as f:
        json.dump(logs, f, indent=4)
        
    # Adaptive logic: If many false positives, increase high_risk threshold
    if len(logs) > 5:
        false_positives = [l for l in logs if l['prediction_prob'] > 0.8 and l['actual_label'] == 'normal']
        if len(false_positives) > 2:
            rules = get_rules()
            rules['high_risk']['threshold'] = min(0.95, rules['high_risk']['threshold'] + 0.05)
            update_rules(rules)
            return "Policy adapted: Increased threshold due to false positives."
            
    return "Feedback recorded."
