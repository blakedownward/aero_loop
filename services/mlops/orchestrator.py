"""Orchestrate the complete MLOps workflow."""

import os
import sys
import argparse
from typing import Optional, Dict, List
from datetime import datetime, timezone

# Add repo root to path for imports
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _repo_root)

from services.remote.download_sessions import download_finished_sessions
from services.remote.deploy_model import deploy_latest_model
from services.mlops.processor import process as process_annotations
from services.mlops.ei_uploader import upload_processed_samples
from services.mlops.ei_trainer import start_training, wait_for_training, get_training_status
from services.mlops.ei_downloader import download_model, get_model_metrics, build_deployment, wait_for_build_completion
from services.mlops.model_evaluator import evaluate_model


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


WORKFLOW_LOG = os.path.join(_repo_root(), 'data', 'workflow_log.json')


def log_workflow_step(step: str, status: str, message: str = None, data: Dict = None):
    """Log a workflow step."""
    import json
    
    if not os.path.isdir(os.path.dirname(WORKFLOW_LOG)):
        os.makedirs(os.path.dirname(WORKFLOW_LOG), exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'step': step,
        'status': status,
        'message': message,
        'data': data or {},
    }
    
    # Append to log file
    log_entries = []
    if os.path.isfile(WORKFLOW_LOG):
        try:
            with open(WORKFLOW_LOG, 'r', encoding='utf-8') as f:
                log_entries = json.load(f)
        except Exception:
            pass
    
    log_entries.append(log_entry)
    
    with open(WORKFLOW_LOG, 'w', encoding='utf-8') as f:
        json.dump(log_entries, f, indent=2)


def step_download_sessions(progress_callback=None) -> Dict[str, any]:
    """Step 1: Download finished sessions from Pi."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 1: Downloading finished sessions from Pi")
        progress_callback("=" * 60)
    
    try:
        result = download_finished_sessions(progress_callback=progress_callback)
        log_workflow_step('download_sessions', 'completed' if result['success'] else 'failed',
                         message=result.get('error'), data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('download_sessions', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_process_annotations(progress_callback=None) -> Dict[str, any]:
    """Step 2: Process annotated files (trim, etc.)."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 2: Processing annotated files")
        progress_callback("=" * 60)
    
    try:
        process_annotations()
        log_workflow_step('process_annotations', 'completed')
        return {'success': True}
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('process_annotations', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_upload_to_ei(progress_callback=None) -> Dict[str, any]:
    """Step 3: Upload processed samples to Edge Impulse."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 3: Uploading to Edge Impulse")
        progress_callback("=" * 60)
    
    try:
        result = upload_processed_samples(progress_callback=progress_callback)
        log_workflow_step('upload_to_ei', 'completed' if result['success'] else 'failed',
                         message=result.get('error'), data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('upload_to_ei', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_train_model(wait: bool = False, progress_callback=None) -> Dict[str, any]:
    """Step 4: Start training on Edge Impulse."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 4: Training model on Edge Impulse")
        progress_callback("=" * 60)
    
    try:
        result = start_training(progress_callback=progress_callback)
        
        if result['success'] and wait:
            if progress_callback:
                progress_callback("Waiting for training to complete...")
            result = wait_for_training(result['job_id'], progress_callback=progress_callback)
        
        log_workflow_step('train_model', 'completed' if result['success'] else 'failed',
                         message=result.get('error'), data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('train_model', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_evaluate_model(progress_callback=None) -> Dict[str, any]:
    """Step 5a: Evaluate model performance (before building/downloading)."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 5: Evaluating model performance")
        progress_callback("=" * 60)
    
    try:
        # Get evaluation metrics from Edge Impulse (no build needed)
        if progress_callback:
            progress_callback("Fetching evaluation metrics from Edge Impulse...")
        
        model_metrics = get_model_metrics()
        
        if not model_metrics:
            return {
                'success': False,
                'error': 'Could not retrieve model metrics from Edge Impulse',
            }
        
        # Evaluate model against best model
        eval_result = evaluate_model(model_metrics)
        
        result = {
            'success': True,
            'evaluation': eval_result,
            'model_metrics': model_metrics,
            'should_build': eval_result.get('is_better', False),
            'should_deploy': eval_result.get('is_better', False),
        }
        
        if progress_callback:
            if eval_result.get('is_better', False):
                progress_callback(f"✓ Model improved! {eval_result.get('reason', '')}")
                progress_callback(f"  New accuracy: {eval_result.get('new_metrics', {}).get('accuracy', 0):.4f}")
                if eval_result.get('best_metrics'):
                    progress_callback(f"  Previous best: {eval_result.get('best_metrics', {}).get('accuracy', 0):.4f}")
            else:
                progress_callback(f"✗ Model did not improve: {eval_result.get('reason', '')}")
                progress_callback(f"  Current accuracy: {eval_result.get('new_metrics', {}).get('accuracy', 0):.4f}")
                if eval_result.get('best_metrics'):
                    progress_callback(f"  Best accuracy: {eval_result.get('best_metrics', {}).get('accuracy', 0):.4f}")
        
        log_workflow_step('evaluate_model', 'completed', data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('evaluate_model', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_build_and_download(deployment_type: str = 'zip', model_type: str = 'float32', 
                           engine: str = 'tflite', wait_for_build: bool = True,
                           progress_callback=None) -> Dict[str, any]:
    """Step 5b: Build and download model (only if evaluation showed improvement)."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 6: Building and downloading model")
        progress_callback("=" * 60)
    
    try:
        # Build deployment
        if progress_callback:
            progress_callback(f"Building deployment ({deployment_type}, {model_type}, {engine})...")
        
        build_result = build_deployment(
            deployment_type=deployment_type,
            model_type=model_type,
            engine=engine,
            progress_callback=progress_callback
        )
        
        if not build_result['success']:
            return build_result
        
        job_id = build_result['job_id']
        
        # Wait for build to complete if requested
        if wait_for_build:
            if progress_callback:
                progress_callback(f"Waiting for build job {job_id} to complete...")
            
            build_status = wait_for_build_completion(job_id, progress_callback=progress_callback)
            
            if build_status['status'] != 'completed':
                return {
                    'success': False,
                    'error': f"Build job {job_id} did not complete successfully (status: {build_status['status']})",
                    'build_status': build_status,
                }
        
        # Download model
        if progress_callback:
            progress_callback("Downloading model...")
        
        download_result = download_model(
            deployment_type=deployment_type,
            model_type=model_type,
            engine=engine,
            progress_callback=progress_callback
        )
        
        if not download_result['success']:
            return download_result
        
        result = {
            'success': True,
            'build': build_result,
            'download': download_result,
        }
        
        if progress_callback:
            progress_callback(f"✓ Model downloaded: {download_result.get('model_filename')}")
        
        log_workflow_step('build_and_download', 'completed', data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('build_and_download', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def step_deploy_model(progress_callback=None) -> Dict[str, any]:
    """Step 6: Deploy improved model to Pi."""
    if progress_callback:
        progress_callback("=" * 60)
        progress_callback("Step 6: Deploying model to Pi")
        progress_callback("=" * 60)
    
    try:
        result = deploy_latest_model(progress_callback=progress_callback)
        log_workflow_step('deploy_model', 'completed' if result['success'] else 'failed',
                         message=result.get('error'), data=result)
        return result
    except Exception as e:
        error_msg = str(e)
        log_workflow_step('deploy_model', 'error', message=error_msg)
        return {'success': False, 'error': error_msg}


def run_workflow(steps: List[str] = None, wait_for_training: bool = False, 
                progress_callback=None) -> Dict[str, any]:
    """Run the complete MLOps workflow.
    
    Args:
        steps: List of steps to run (if None, runs all)
        wait_for_training: Whether to wait for training to complete
        progress_callback: Optional callback for progress updates
    
    Returns: Dictionary with workflow results
    """
    all_steps = [
        'download_sessions',
        'process_annotations',
        'upload_to_ei',
        'train_model',
        'evaluate_model',
        'build_and_download',
        'deploy_model',
    ]
    
    if steps is None:
        steps = all_steps.copy()
    else:
        steps = steps.copy()  # Work with a copy to avoid modification issues
    
    results = {}
    skip_deployment = False
    
    for step in steps:
        if step not in all_steps:
            if progress_callback:
                progress_callback(f"Warning: Unknown step: {step}")
            continue
        
        # Skip deployment if model didn't improve
        if step == 'deploy_model' and skip_deployment:
            results[step] = {
                'success': False,
                'error': 'Model did not improve, skipping deployment',
                'skipped': True,
            }
            if progress_callback:
                progress_callback("Skipping deployment: model did not improve")
            continue
        
        if step == 'download_sessions':
            results[step] = step_download_sessions(progress_callback)
        elif step == 'process_annotations':
            results[step] = step_process_annotations(progress_callback)
        elif step == 'upload_to_ei':
            results[step] = step_upload_to_ei(progress_callback)
        elif step == 'train_model':
            results[step] = step_train_model(wait=wait_for_training, progress_callback=progress_callback)
        elif step == 'evaluate_model':
            results[step] = step_evaluate_model(progress_callback)
            # Check if model improved - if not, skip build/download and deployment
            if results[step].get('success') and not results[step].get('should_build', False):
                skip_deployment = True
                if progress_callback:
                    progress_callback("\nModel did not improve. Will skip build/download and deployment.")
        elif step == 'build_and_download':
            # Only build/download if evaluation showed improvement
            if 'evaluate_model' in results:
                eval_result = results['evaluate_model']
                if eval_result.get('should_build', False):
                    results[step] = step_build_and_download(
                        wait_for_build=True,
                        progress_callback=progress_callback
                    )
                else:
                    results[step] = {
                        'success': False,
                        'error': 'Model did not improve, skipping build and download',
                        'skipped': True,
                    }
                    if progress_callback:
                        progress_callback("Skipping build/download: model did not improve")
            else:
                # Build/download anyway if evaluation step wasn't run
                results[step] = step_build_and_download(
                    wait_for_build=True,
                    progress_callback=progress_callback
                )
        elif step == 'deploy_model':
            # Only deploy if previous evaluation showed improvement
            if 'evaluate_model' in results:
                eval_result = results['evaluate_model']
                if eval_result.get('should_deploy', False):
                    results[step] = step_deploy_model(progress_callback)
                else:
                    results[step] = {
                        'success': False,
                        'error': 'Model did not improve, skipping deployment',
                        'skipped': True,
                    }
                    if progress_callback:
                        progress_callback("Skipping deployment: model did not improve")
            else:
                # Deploy anyway if evaluation step wasn't run
                results[step] = step_deploy_model(progress_callback)
        
        # Check if step failed
        if not results[step].get('success', False) and not results[step].get('skipped', False):
            if progress_callback:
                progress_callback(f"\nWorkflow stopped at step '{step}' due to error")
            break
    
    # Summary
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    
    summary = {
        'success': success_count == total_count,
        'steps_completed': success_count,
        'steps_total': total_count,
        'results': results,
    }
    
    log_workflow_step('workflow_complete', 'completed' if summary['success'] else 'partial',
                     data=summary)
    
    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='AeroLoop MLOps Orchestrator')
    parser.add_argument('--steps', nargs='+', 
                       choices=['download_sessions', 'process_annotations', 'upload_to_ei',
                               'train_model', 'evaluate_model', 'build_and_download', 'deploy_model'],
                       help='Steps to run (default: all)')
    parser.add_argument('--wait-training', action='store_true',
                       help='Wait for training to complete')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip annotation step (assumes already done)')
    
    args = parser.parse_args()
    
    steps = args.steps
    if args.skip_annotation and steps:
        steps = [s for s in steps if s != 'process_annotations']
    
    def print_progress(msg):
        print(msg)
    
    result = run_workflow(steps=steps, wait_for_training=args.wait_training,
                         progress_callback=print_progress)
    
    print("\n" + "=" * 60)
    print("Workflow Summary")
    print("=" * 60)
    print(f"Steps completed: {result['steps_completed']}/{result['steps_total']}")
    print(f"Overall success: {result['success']}")
    
    if not result['success']:
        print("\nFailed steps:")
        for step, step_result in result['results'].items():
            if not step_result.get('success', False):
                print(f"  - {step}: {step_result.get('error', 'Unknown error')}")
    
    return 0 if result['success'] else 1


if __name__ == '__main__':
    sys.exit(main())

