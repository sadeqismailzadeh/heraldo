import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import os
import shutil

# ============= CONFIGURATION =============
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train", "PPO_1")
LOG_DIR = log_dir
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_plots")

# TensorBoard Smoothing Factor (0.0 to 0.999)
# 0.0 = No smoothing
# 0.6 = Typical TensorBoard default
# 0.9 = Very smooth
SMOOTHING_WEIGHT = 0.6

TAGS_TO_EXTRACT = [
    'all' 
    # 'rollout/ep_rew_mean',
    # 'train/loss',
]

LIST_ONLY = False

# Plot Settings
FIG_SIZE = (10, 6)    # Width, Height in inches
DPI = 300             # Resolution
GRID_ALPHA = 0.3      # Opacity of the grid lines
RAW_DATA_ALPHA = 0.3  # Opacity of the unsmoothed background line
# =========================================

def smooth_data(values, weight):
    """
    Apply exponential moving average smoothing (matches TensorBoard logic).
    Using Pandas ewm (Exponential Weighted Functions).
    """
    if weight <= 0:
        return values
    
    # TensorBoard uses exponential moving average. 
    # alpha = 1 - weight
    series = pd.Series(values)
    smoothed = series.ewm(alpha=(1 - weight)).mean()
    return smoothed.tolist()

def generate_plots(log_dir, tags, output_dir):
    """
    Extract metrics and generate smoothed plots.
    """
    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    
    available_tags = ea.Tags()['scalars']
    
    # Filter tags
    if not tags or tags == ['all']:
        tags = available_tags
        print("Processing all available metrics...")
    else:
        tags = [tag for tag in tags if tag in available_tags]
    
    if not tags:
        print("No valid tags found to plot.")
        return

    # Create output directory
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' exists. Saving new files there.")
    else:
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"\nGenerating {len(tags)} figures with smoothing {SMOOTHING_WEIGHT}...")

    for i, tag in enumerate(tags):
        # Extract data
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        # Calculate smoothing
        smoothed_values = smooth_data(values, SMOOTHING_WEIGHT)
        
        # Setup Figure
        plt.figure(figsize=FIG_SIZE)
        
        # 1. Plot Raw Data (Faint background line)
        plt.plot(steps, values, color='#1f77b4', alpha=RAW_DATA_ALPHA, linewidth=1, label='Original')
        
        # 2. Plot Smoothed Data (Solid foreground line)
        plt.plot(steps, smoothed_values, color='#1f77b4', linewidth=2, label=f'Smoothed ({SMOOTHING_WEIGHT})')
        
        # Formatting
        plt.title(tag, fontsize=14)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=GRID_ALPHA)
        plt.legend()
        
        # Sanitize filename (replace / with _)
        safe_name = tag.replace('/', '_').replace(' ', '_')
        file_path = os.path.join(output_dir, f"{safe_name}.png")
        
        # Save and close
        plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
        plt.close() # Close memory reference
        
        print(f"[{i+1}/{len(tags)}] Saved: {file_path}")

    print(f"\n✓ Done! Check the folder: {output_dir}")

def find_event_files(log_dir):
    """Recursively find TensorBoard event files."""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    return event_files

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' does not exist")
        return
    
    event_files = find_event_files(LOG_DIR)
    if not event_files:
        print(f"Error: No TensorBoard event files found in '{LOG_DIR}'")
        return
    
    # Use the most recent event file or dir
    log_path = LOG_DIR if len(event_files) == 1 else os.path.dirname(event_files[-1])
    
    if LIST_ONLY:
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        print(f"\nAvailable metrics:")
        for tag in sorted(ea.Tags()['scalars']):
            print(f"  - {tag}")
        return
    
    generate_plots(log_path, TAGS_TO_EXTRACT, OUTPUT_DIR)

if __name__ == '__main__':
    main()