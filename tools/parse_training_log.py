#!/usr/bin/env python3
"""
Script to parse training log files and extract metrics, losses, and generate plots.

Usage: python parse_training_log.py <log_file.err or .log>
"""

import sys
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import ast
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from tools/ to project root
sys.path.insert(0, project_root)

from stereo.utils import common_utils
import train as train_utils


def parse_tensor_dict(tensor_str):
    """Parse a string containing tensor values into a regular dict."""
    try:
        # Replace tensor() with float values, handle both regular and scientific notation
        cleaned = re.sub(r'tensor\(([\d\.-e]+)\)', r'\1', tensor_str)
        # Use ast.literal_eval to safely evaluate the dict
        result = ast.literal_eval(cleaned)
        # Convert all values to float and handle NaN
        for key, value in result.items():
            try:
                result[key] = float(value)
                if np.isnan(result[key]):
                    result[key] = float('nan')
            except:
                result[key] = -1
        return result
    except Exception as e:
        print(f"Warning: Could not parse tensor dict: {tensor_str}")
        return {}


def extract_epoch_from_line(line, pattern_type="evaluation"):
    """Extract epoch number from different line types."""
    if pattern_type == "evaluation":
        # Pattern: "Epoch 40 metrics:"
        match = re.search(r'Epoch (\d+) metrics:', line)
    elif pattern_type == "testing":
        # Pattern: "Testing: Epoch 49 metrics:"
        match = re.search(r'Testing: Epoch (\d+) metrics:', line)
    elif pattern_type == "training":
        # Pattern: "Training Epoch:49/50" or "Training Epoch: 0/50"
        match = re.search(r'Training Epoch:?\s*(\d+)/\d+', line)
    
    return int(match.group(1)) if match else None


def parse_training_line(line):
    """Parse training progress line to extract loss and learning rate."""
    # Pattern: Loss:1.81540(1.88145) LR:4.3183e-07
    # Extract the first loss value (not the one in parentheses)
    loss_match = re.search(r'Loss:([\d\.-]+(?:e[+-]?\d+)?)\(', line)
    lr_match = re.search(r'LR:([\d\.-]+(?:e[+-]?\d+)?)', line)
    
    loss = None
    lr = None
    
    if loss_match:
        try:
            loss = float(loss_match.group(1))
            if np.isnan(loss):
                loss = float('nan')
        except Exception as e:
            print(f"Error parsing loss: {e}")
            print(f"Original line: {line}")
            print(f"Loss match: {loss_match.group(1)}")
            loss = -1
    
    if lr_match:
        try:
            lr = float(lr_match.group(1))
            if np.isnan(lr):
                lr = float('nan')
        except Exception as e:
            print(f"Error parsing lr: {e}")
            print(f"Original line: {line}")
            print(f"LR match: {lr_match.group(1)}")
            lr = -1
    
    return loss, lr


def extract_job_and_experiment_id(filename):
    """Extract job ID and experiment ID from filename."""
    # Pattern: 59386399_03_mobile_one_s0_local_from_scratch.err
    # Job ID: 59386399 (before first _)
    # Experiment ID: 03 (between first and second _)
    
    parts = filename.split('_')
    job_id = parts[0] if len(parts) > 0 else "unknown"
    experiment_id = parts[1] if len(parts) > 1 else "unknown"
    
    return job_id, experiment_id


def parse_log_file(filepath):
    """Parse the entire log file and extract all relevant information."""
    data = {
        'epochs': {},
        'model_info': {
            'gflops': -1,
            'parameters_m': -1
        },
        'final_testing': {},
        'best_epoch': {
            'idx': -1,
            'epe': -1
        }
    }
    
    # Track training data by epoch
    training_data = defaultdict(lambda: {'losses': [], 'lrs': []})
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse evaluation metrics (during training)
            if 'Epoch' in line and 'metrics:' in line and 'Testing:' not in line:
                epoch = extract_epoch_from_line(line, "evaluation")
                if epoch is not None:
                    # Extract metrics dict
                    metrics_match = re.search(r"metrics: ({.*})", line)
                    if metrics_match:
                        metrics_dict = parse_tensor_dict(metrics_match.group(1))
                        if epoch not in data['epochs']:
                            data['epochs'][epoch] = {}
                        data['epochs'][epoch]['evaluation'] = metrics_dict
            
            # Parse final testing metrics
            elif 'Testing: Epoch' in line and 'metrics:' in line:
                epoch = extract_epoch_from_line(line, "testing")
                if epoch is not None:
                    metrics_match = re.search(r"metrics: ({.*})", line)
                    if metrics_match:
                        metrics_dict = parse_tensor_dict(metrics_match.group(1))
                        data['final_testing'] = {
                            'epoch': epoch,
                            'metrics': metrics_dict
                        }
            
            # Parse training progress lines - handle both formats
            elif ('Training Epoch:' in line or 'Training Epoch: ' in line) and 'Loss:' in line and 'LR:' in line:
                epoch = extract_epoch_from_line(line, "training")
                if epoch is not None:
                    loss, lr = parse_training_line(line)
                    if loss is not None:
                        training_data[epoch]['losses'].append(loss)
                    if lr is not None:
                        training_data[epoch]['lrs'].append(lr)
            
            # Parse model information
            elif 'Number of calculates:' in line:
                match = re.search(r'Number of calculates: ([\d\.]+) GFlops', line)
                if match:
                    data['model_info']['gflops'] = float(match.group(1))
            
            elif 'Number of parameters:' in line:
                match = re.search(r'Number of parameters: ([\d\.]+) M', line)
                if match:
                    data['model_info']['parameters_m'] = float(match.group(1))
            
            # Parse best epoch information
            elif 'Final best epoch idx:' in line:
                match = re.search(r'Final best epoch idx: (\d+)', line)
                if match:
                    data['best_epoch']['idx'] = int(match.group(1))
            
            elif 'Final best epoch epe:' in line:
                match = re.search(r'Final best epoch epe: ([\d\.]+)', line)
                if match:
                    data['best_epoch']['epe'] = float(match.group(1))
    
    # Calculate average loss and LR for each epoch
    for epoch, train_data in training_data.items():
        if epoch not in data['epochs']:
            data['epochs'][epoch] = {}
        
        # Calculate averages
        avg_loss = np.mean(train_data['losses']) if train_data['losses'] else -1
        avg_lr = np.mean(train_data['lrs']) if train_data['lrs'] else -1
        
        # Handle NaN values
        avg_loss = avg_loss if not np.isnan(avg_loss) else float('nan')
        avg_lr = avg_lr if not np.isnan(avg_lr) else float('nan')
        
        data['epochs'][epoch]['training'] = {
            'avg_loss': avg_loss,
            'avg_lr': avg_lr,
            'num_samples': len(train_data['losses'])
        }
    
    return data


def create_plots(data, output_dir, job_id, experiment_id):
    """Create matplotlib plots for the extracted data."""
    epochs = sorted(data['epochs'].keys())
    
    # Extract training data
    train_losses = []
    train_lrs = []
    eval_epes = []
    eval_epochs = []
    
    for epoch in epochs:
        epoch_data = data['epochs'][epoch]
        
        # Training data
        if 'training' in epoch_data:
            train_losses.append(epoch_data['training']['avg_loss'])
            train_lrs.append(epoch_data['training']['avg_lr'])
        else:
            train_losses.append(np.nan)
            train_lrs.append(np.nan)
        
        # Evaluation data
        if 'evaluation' in epoch_data:
            eval_epochs.append(epoch)
            eval_epes.append(epoch_data['evaluation'].get('epe', np.nan))
    
    # Plot 1: Training Loss and Learning Rate
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Loss on left axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis limit for loss to 1.5 times the second epoch value (epoch 1)
    if len(train_losses) > 1 and not np.isnan(train_losses[1]) and train_losses[1] > 0:
        max_loss_display = 1.5 * train_losses[1]
        ax1.set_ylim(bottom=0, top=max_loss_display)
    
    # Learning rate on right axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(epochs, train_lrs, color=color, marker='s', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')  # Log scale for LR
    
    plt.title(f'Training Progress: Loss and Learning Rate (Job {job_id}, Exp {experiment_id})')
    plt.tight_layout()
    plt.savefig(output_dir / f'{job_id}_{experiment_id}_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Evaluation EPE
    if eval_epes:
        plt.figure(figsize=(10, 6))
        plt.plot(eval_epochs, eval_epes, color='tab:green', marker='o', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('EPE (End Point Error)')
        plt.title(f'Evaluation EPE Over Training (Job {job_id}, Exp {experiment_id})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'{job_id}_{experiment_id}_evaluation_epe.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")
    print(f"  - {job_id}_{experiment_id}_training_progress.png")
    if eval_epes:
        print(f"  - {job_id}_{experiment_id}_evaluation_epe.png")


def parse_tensorrt_json(json_filepath):
    """
    Parse TensorRT JSON file and extract benchmark metrics.
    
    Args:
        json_filepath (str or Path): Path to the TensorRT JSON file
        
    Returns:
        dict: Dictionary containing extracted TensorRT metrics
    """
    tensorrt_data = {
        'inference_throughput': -1,
        'trt_epe': -1,
        'engine_footprint_mb': -1,
        'mem_mb_mean': -1
    }
    
    json_path = Path(json_filepath)
    if not json_path.exists():
        print(f"Warning: TensorRT JSON file not found: {json_path}")
        return tensorrt_data
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract inference throughput
        if 'inference_benchmark' in data:
            onnx_trt = data['inference_benchmark'].get('Onnx_trt_inference', {})
            tensorrt_data['inference_throughput'] = onnx_trt.get('inference', -1)
        
        # Extract TensorRT EPE
        if 'test_accuracy_metrics_trt' in data:
            trt_metrics = data['test_accuracy_metrics_trt']
            tensorrt_data['trt_epe'] = trt_metrics.get('epe', -1)
        
        # Extract memory information
        if 'memory' in data:
            memory_info = data['memory']
            if 'engine_footprint' in memory_info:
                tensorrt_data['engine_footprint_mb'] = memory_info['engine_footprint'].get('total_model_memory_mb', -1)
            if 'inference_memory_stats' in memory_info:
                tensorrt_data['mem_mb_mean'] = memory_info['inference_memory_stats'].get('mem_mb_mean', -1)
        
        print(f"Successfully parsed TensorRT JSON: {json_path}")
        
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse TensorRT JSON file {json_path}: {e}")
    except Exception as e:
        print(f"Warning: Error reading TensorRT JSON file {json_path}: {e}")
    
    return tensorrt_data


def do_log_parsing(filename, json_filename=None):
    """
    Parse training log file and extract metrics, losses, and generate plots.
    
    Args:
        filename (str or Path): Path to the .err or .log log file
        
    Returns:
        str: Summary text containing parsed information
    """
    log_path = Path(filename)
    if not log_path.exists():
        raise FileNotFoundError(f"File {log_path} does not exist")
    
    print(f"Parsing log file: {log_path}")
    
    # Extract job ID and experiment ID from filename
    job_id, experiment_id = extract_job_and_experiment_id(log_path.name)
    print(f"Detected Job ID: {job_id}, Experiment ID: {experiment_id}")
    
    # Parse the log file
    data = parse_log_file(log_path)
    
    # Parse TensorRT JSON file if provided
    tensorrt_data = None
    if json_filename is not None:
        print(f"Parsing TensorRT JSON file: {json_filename}")
        tensorrt_data = parse_tensorrt_json(json_filename)
    
    # Add job and experiment info to data
    data['job_info'] = {
        'job_id': job_id,
        'experiment_id': experiment_id,
        'filename': log_path.name
    }
    
    # Add TensorRT data to main data structure if available
    if tensorrt_data is not None:
        data['tensorrt_benchmark'] = tensorrt_data
    
    # Create output directory and files
    output_dir = log_path.parent
    base_name = log_path.stem  # filename without extension
    
    # Save JSON data
    json_path = output_dir / f"{base_name}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)  # default=str handles NaN values
    
    print(f"Data saved to: {json_path}")
    
    # Create plots
    create_plots(data, output_dir, job_id, experiment_id)
    
    # Build summary text
    summary_lines = []
    summary_lines.append("="*50)
    summary_lines.append("SUMMARY")
    summary_lines.append("="*50)
    summary_lines.append(f"Job ID: {job_id}")
    summary_lines.append(f"Experiment ID: {experiment_id}")
    
    if data['model_info']['gflops'] != -1:
        summary_lines.append(f"Model GFLOPs: {data['model_info']['gflops']:.2f}")
    if data['model_info']['parameters_m'] != -1:
        summary_lines.append(f"Model Parameters: {data['model_info']['parameters_m']:.2f}M")
    
    summary_lines.append(f"Epochs parsed: {len(data['epochs'])}")
    
    # Show evaluation EPE progress (best and final evaluation during training)
    epochs = sorted(data['epochs'].keys())
    eval_epes = []
    for epoch in epochs:
        if 'evaluation' in data['epochs'][epoch] and 'epe' in data['epochs'][epoch]['evaluation']:
            eval_epes.append(data['epochs'][epoch]['evaluation']['epe'])
    
    if eval_epes:
        best_eval_epe = min(eval_epes)
        final_eval_epe = eval_epes[-1] if eval_epes else None
        summary_lines.append(f"Best evaluation EPE: {best_eval_epe:.4f}")
        summary_lines.append(f"Final evaluation EPE: {final_eval_epe:.4f}")
    
    # Show best epoch information (if available)
    if data['best_epoch']['idx'] != -1 and data['best_epoch']['epe'] != -1:
        summary_lines.append(f"Final best epoch idx: {data['best_epoch']['idx']}")
        summary_lines.append(f"Final best epoch epe: {data['best_epoch']['epe']:.4f}")
    
    # Show final testing EPE (if available)
    if data['final_testing']:
        summary_lines.append(f"Final testing epoch: {data['final_testing']['epoch']}")
        final_metrics = data['final_testing']['metrics']
        if 'epe' in final_metrics:
            summary_lines.append(f"Final testing EPE: {final_metrics['epe']:.4f}")
        if 'd1_all' in final_metrics:
            summary_lines.append(f"Final testing D1-all: {final_metrics['d1_all']:.4f}%")
    
    # Show training progress
    if epochs:
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        
        if 'training' in data['epochs'][first_epoch]:
            first_loss = data['epochs'][first_epoch]['training']['avg_loss']
            summary_lines.append(f"First epoch training loss: {first_loss:.4f}")
        
        if 'training' in data['epochs'][last_epoch]:
            last_loss = data['epochs'][last_epoch]['training']['avg_loss']
            summary_lines.append(f"Last epoch training loss: {last_loss:.4f}")
    
    # Show TensorRT benchmark information (if available)
    if 'tensorrt_benchmark' in data:
        trt_data = data['tensorrt_benchmark']
        summary_lines.append("")  # Add blank line for separation
        summary_lines.append("TensorRT Benchmark:")
        
        if 'inference_throughput' in trt_data:
            summary_lines.append(f"  Inference throughput: {trt_data['inference_throughput']:.2f} inferences/sec")
        else:
            summary_lines.append(f"  Inference throughput: -1")
        
        if 'trt_epe' in trt_data:
            summary_lines.append(f"  TensorRT EPE: {trt_data['trt_epe']:.4f}")
        else:
            summary_lines.append(f"  TensorRT EPE: -1")

        if 'engine_footprint_mb' in trt_data:
            summary_lines.append(f"  Engine footprint: {trt_data['engine_footprint_mb']:.2f} MB")
        else:
            summary_lines.append(f"  Engine footprint: -1")

        if 'mem_mb_mean' in trt_data:
            summary_lines.append(f"  Memory usage (mean): {trt_data['mem_mb_mean']:.2f} MB")
        else:
            summary_lines.append(f"  Memory usage (mean): -1")
    
    # Join all summary lines
    summary_text = "\n".join(summary_lines)
    
    # Print summary to console
    print("\n" + summary_text)
    
    # Save summary to txt file
    summary_path = output_dir / f"{base_name}_experiment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Summary saved to: {summary_path}")
    
    return summary_text


def parse_log_filename(args):
    """
    Parse log filename from args and cfgs configuration.
    
    Args:
        args: Parsed arguments from parse_config()
    
    This function will be implemented by the user.
    """
    return os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}.log")

def parse_tensorRT_json_filename(args):
    return os.path.join(args.output_dir, f"{args.slurm_job_id}_{args.experiment_id}_tensorRT_summary.json")

def main():

    print(f"parse_training_log.py: Starting log parsing...")

    # Check number of command line arguments
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        # Single argument - use current behavior
        parser = argparse.ArgumentParser(description='Parse training log file and extract metrics')
        parser.add_argument('log_file', help='Path to the .err or .log log file')
        parser.add_argument('tensorRT_json_file', help='Path to the TensorRT JSON file', default=None)
        args = parser.parse_args()
        
        try:
            print(f"parse_training_log.py: Parsing log file: {args.log_file}")
            print(f"parse_training_log.py: Parsing TensorRT JSON file: {args.tensorRT_json_file}")
            do_log_parsing(args.log_file, args.tensorRT_json_file)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Multiple arguments - the same call as to the train.py - use parse_config()
        print(f"Multiple arguments detected, using parse_config() for configuration parsing")
        args, cfgs = train_utils.parse_config()
        tensorRT_json_file = parse_tensorRT_json_filename(args)
        filename = parse_log_filename(args)
        do_log_parsing(filename, tensorRT_json_file)

if __name__ == "__main__":
    main()
