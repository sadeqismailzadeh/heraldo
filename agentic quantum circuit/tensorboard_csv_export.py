import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os
from pathlib import Path

# ============= CONFIGURATION =============
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Train", "PPO_1")
LOG_DIR = log_dir  # Path to your TensorBoard log directory
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensorboard_data.csv")  # Output CSV file name

# Specify which metrics to extract (use 'all' to extract everything)
TAGS_TO_EXTRACT = [
    'all'
    # 'rollout/ep_rew_mean',
    # 'train/clip_fraction',
    # Add more metrics here as needed:
    # 'train/loss',
    # 'train/policy_loss',
    # 'train/value_loss',
    # 'train/entropy_loss',
]

# Set to True to see all available metrics without extracting
LIST_ONLY = False

# CSV Format Options:
# 'wide' - One column per metric (traditional format, good for spreadsheets)
#   Example: step, rollout/ep_rew_mean, train/clip_fraction
# 'long' - One row per metric-step combination (better for AI analysis)
#   Example: step, metric_name, value
# 'both' - Save both formats with '_wide' and '_long' suffixes
FORMAT = 'long'  # Options: 'wide', 'long', 'both'

# Decimal precision (set lower for cleaner AI input, None for full precision)
DECIMAL_PLACES = 4  # e.g., 0.123456 becomes 0.1235
# =========================================


def extract_tensorboard_data(log_dir, tags, output_file, format_type='both'):
    """
    Extract specified metrics from TensorBoard logs and save to CSV.
    
    Args:
        log_dir: Path to TensorBoard log directory
        tags: List of metric names to extract (e.g., ['rollout/ep_rew_mean', 'train/clip_fraction'])
        output_file: Output CSV file path
        format_type: 'wide', 'long', or 'both'
    """
    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all
        }
    )
    ea.Reload()
    
    # Get available tags
    available_tags = ea.Tags()['scalars']
    print(f"Available metrics in TensorBoard logs:")
    for tag in sorted(available_tags):
        print(f"  - {tag}")
    print()
    
    # If no tags specified, use all available
    if not tags or tags == ['all']:
        tags = available_tags
        print("Extracting all available metrics")
    else:
        # Validate requested tags
        missing_tags = set(tags) - set(available_tags)
        if missing_tags:
            print(f"Warning: The following tags were not found: {missing_tags}")
        tags = [tag for tag in tags if tag in available_tags]
    
    if not tags:
        print("No valid tags to extract!")
        return
    
    print(f"\nExtracting {len(tags)} metrics:")
    for tag in tags:
        print(f"  - {tag}")
    
    # Extract data for each tag
    data = {}
    all_steps = set()
    
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        all_steps.update(steps)
        data[tag] = dict(zip(steps, values))
    
    # Create wide format DataFrame
    all_steps = sorted(all_steps)
    df_data = {'step': all_steps}
    
    for tag in tags:
        df_data[tag] = [data[tag].get(step, None) for step in all_steps]
    
    df_wide = pd.DataFrame(df_data)
    
    # Round values if specified
    if DECIMAL_PLACES is not None:
        for col in df_wide.columns:
            if col != 'step':
                df_wide[col] = df_wide[col].round(DECIMAL_PLACES)
    
    # Save based on format type
    base_name = output_file.rsplit('.', 1)[0]
    ext = output_file.rsplit('.', 1)[1] if '.' in output_file else 'csv'
    
    if format_type in ['wide', 'both']:
        wide_file = f"{base_name}_wide.{ext}" if format_type == 'both' else output_file
        df_wide.to_csv(wide_file, index=False)
        print(f"\n✓ Wide format: {len(df_wide)} rows × {len(df_wide.columns)} columns → {wide_file}")
    
    if format_type in ['long', 'both']:
        # Convert to long format
        df_long = df_wide.melt(
            id_vars=['step'],
            var_name='metric',
            value_name='value'
        )
        # Remove rows with missing values
        df_long = df_long.dropna()
        
        # Round values if specified
        if DECIMAL_PLACES is not None:
            df_long['value'] = df_long['value'].round(DECIMAL_PLACES)
        
        # Sort by step then metric
        df_long = df_long.sort_values(['step', 'metric']).reset_index(drop=True)
        
        long_file = f"{base_name}_long.{ext}" if format_type == 'both' else output_file
        df_long.to_csv(long_file, index=False)
        print(f"✓ Long format: {len(df_long)} rows × {len(df_long.columns)} columns → {long_file}")
    
    print(f"\n📊 Summary:")
    print(f"   Steps: {len(all_steps)}")
    print(f"   Metrics: {len(tags)}")


def find_event_files(log_dir):
    """Recursively find TensorBoard event files in directory."""
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    return event_files


def main():
    # Check if log directory exists
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' does not exist")
        return
    
    # Find event files
    event_files = find_event_files(LOG_DIR)
    if not event_files:
        print(f"Error: No TensorBoard event files found in '{LOG_DIR}'")
        return
    
    print(f"Found {len(event_files)} event file(s)")
    
    # Use the most recent event file or the directory if there's only one
    log_path = LOG_DIR if len(event_files) == 1 else os.path.dirname(event_files[-1])
    
    if LIST_ONLY:
        # Just list available metrics
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        available_tags = ea.Tags()['scalars']
        print(f"\nAvailable metrics ({len(available_tags)}):")
        for tag in sorted(available_tags):
            print(f"  - {tag}")
        return
    
    # Extract data
    extract_tensorboard_data(log_path, TAGS_TO_EXTRACT, OUTPUT_FILE, FORMAT)


if __name__ == '__main__':
    main()