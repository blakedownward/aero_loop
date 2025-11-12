"""Evaluate and compare model performance."""

import os
import json
from typing import Optional, Dict, Tuple
from datetime import datetime


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


MODELS_PATH = os.path.join(_repo_root(), 'models')
BEST_MODEL_LOG = os.path.join(_repo_root(), 'data', 'best_model.json')


def load_best_model_log() -> Dict:
    """Load the best model log."""
    if os.path.isfile(BEST_MODEL_LOG):
        try:
            with open(BEST_MODEL_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_best_model_log(log: Dict):
    """Save the best model log."""
    os.makedirs(os.path.dirname(BEST_MODEL_LOG), exist_ok=True)
    with open(BEST_MODEL_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)


def extract_metrics(model_metrics: Dict) -> Dict[str, float]:
    """Extract key metrics from Edge Impulse model metrics.
    
    Returns dict with: accuracy, loss, f1_score, precision, recall
    """
    metrics = {}
    
    # Try to extract from various possible structures
    if isinstance(model_metrics, dict):
        # New API format: accuracy is nested in 'accuracy' -> 'raw'
        if 'accuracy' in model_metrics:
            acc_data = model_metrics['accuracy']
            if isinstance(acc_data, dict):
                metrics['accuracy'] = acc_data.get('raw', acc_data.get('value', 0.0))
            else:
                metrics['accuracy'] = float(acc_data) if acc_data else 0.0
        
        # Test set accuracy (legacy format)
        if 'test' in model_metrics:
            test_metrics = model_metrics['test']
            if isinstance(test_metrics, dict):
                metrics['accuracy'] = test_metrics.get('accuracy', metrics.get('accuracy', 0.0))
                metrics['loss'] = test_metrics.get('loss', float('inf'))
                metrics['f1_score'] = test_metrics.get('f1_score', 0.0)
                metrics['precision'] = test_metrics.get('precision', 0.0)
                metrics['recall'] = test_metrics.get('recall', 0.0)
        
        # Direct metrics (legacy format)
        if 'accuracy' not in metrics and 'accuracy' in model_metrics:
            acc_val = model_metrics['accuracy']
            if isinstance(acc_val, (int, float)):
                metrics['accuracy'] = float(acc_val)
            else:
                metrics['accuracy'] = 0.0
        
        if 'loss' in model_metrics:
            loss_val = model_metrics['loss']
            if isinstance(loss_val, (int, float)):
                metrics['loss'] = float(loss_val)
            else:
                metrics['loss'] = float('inf')
        
        if 'f1_score' in model_metrics:
            metrics['f1_score'] = model_metrics.get('f1_score', 0.0)
    
    # Defaults
    if 'accuracy' not in metrics:
        metrics['accuracy'] = 0.0
    if 'loss' not in metrics:
        metrics['loss'] = float('inf')
    if 'f1_score' not in metrics:
        metrics['f1_score'] = 0.0
    if 'precision' not in metrics:
        metrics['precision'] = 0.0
    if 'recall' not in metrics:
        metrics['recall'] = 0.0
    
    return metrics


def compare_models(new_metrics: Dict, best_metrics: Dict) -> Tuple[bool, str]:
    """Compare new model metrics against best model.
    
    Returns: (is_better, reason)
    """
    new_acc = new_metrics.get('accuracy', 0.0)
    new_loss = new_metrics.get('loss', float('inf'))
    new_f1 = new_metrics.get('f1_score', 0.0)
    
    best_acc = best_metrics.get('accuracy', 0.0)
    best_loss = best_metrics.get('loss', float('inf'))
    best_f1 = best_metrics.get('f1_score', 0.0)
    
    # Primary: Check if accuracy improved
    if new_acc > best_acc:
        return True, f"Accuracy improved: {new_acc:.4f} > {best_acc:.4f}"
    
    # Secondary: Check if loss improved (lower is better)
    if new_acc == best_acc and new_loss < best_loss:
        return True, f"Loss improved: {new_loss:.4f} < {best_loss:.4f}"
    
    # Tertiary: Check if F1 score improved
    if new_acc == best_acc and new_loss == best_loss and new_f1 > best_f1:
        return True, f"F1 score improved: {new_f1:.4f} > {best_f1:.4f}"
    
    return False, f"No improvement (acc: {new_acc:.4f} <= {best_acc:.4f}, loss: {new_loss:.4f} >= {best_loss:.4f})"


def evaluate_model(model_metrics: Dict, model_filename: str = None) -> Dict[str, any]:
    """Evaluate a model and determine if it's better than the current best.
    
    Args:
        model_metrics: Model metrics from Edge Impulse
        model_filename: Optional filename of the model file
    
    Returns: Dictionary with evaluation results
    """
    extracted = extract_metrics(model_metrics)
    
    best_log = load_best_model_log()
    
    # If no best model yet, this is the best
    if not best_log or 'metrics' not in best_log:
        best_log = {
            'model_filename': model_filename,
            'metrics': extracted,
            'evaluated_at': datetime.utcnow().isoformat() + 'Z',
        }
        save_best_model_log(best_log)
        
        return {
            'is_better': True,
            'is_best': True,
            'reason': 'First model evaluated',
            'new_metrics': extracted,
            'best_metrics': extracted,
        }
    
    # Compare with best
    best_metrics = best_log.get('metrics', {})
    is_better, reason = compare_models(extracted, best_metrics)
    
    if is_better:
        # Update best model
        best_log['model_filename'] = model_filename
        best_log['metrics'] = extracted
        best_log['evaluated_at'] = datetime.utcnow().isoformat() + 'Z'
        save_best_model_log(best_log)
    
    return {
        'is_better': is_better,
        'is_best': is_better,
        'reason': reason,
        'new_metrics': extracted,
        'best_metrics': best_metrics,
        'best_model_filename': best_log.get('model_filename'),
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_evaluator.py <metrics_json_file>")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    if not os.path.isfile(metrics_file):
        print(f"Error: Metrics file not found: {metrics_file}")
        sys.exit(1)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    result = evaluate_model(metrics)
    
    print(f"Evaluation result: {result['is_better']}")
    print(f"Reason: {result['reason']}")
    print(f"New metrics: {result['new_metrics']}")
    print(f"Best metrics: {result['best_metrics']}")

