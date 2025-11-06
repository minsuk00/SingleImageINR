import copy
import itertools
from datetime import datetime
import yaml
import torch
import wandb
import argparse

# Import the main training function from your train.py file
try:
    from train import main as train_main
except ImportError:
    print("Error: Could not import the 'main' function from 'train.py'.")
    print("Please make sure 'train.py' is in the same directory.")
    exit(1)


# Helper function to set nested dictionary keys using a dot-separated string
def set_nested_key(d, key_str, value):
    """
    Set a value in a nested dictionary using a dot-separated key string.
    Example: set_nested_key(cfg, 'loss.pixel_weight', 0.5)
    """
    keys = key_str.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def run_sweep(wandb_enabled):
    # --- Define Your Parameter Grid ---
    param_grid = {
        "lr": [1e-3],
        # Using a smaller hash size to prevent OOM
        "hash.encoding_config.log2_hashmap_size": [19],
        "loss.pixel_weight": [1.0, 0.5],
        "loss.ssim_weight": [0.0, 0.5],
        "loss.lpips_weight": [0.0, 0.5],
        "loss.chess_weight": [0.5, 1.0],
    }

    # --- Load Base Configuration ---
    base_config_path = "/home/sincheol/SingleImageINR/config/config.yaml"
    try:
        with open(base_config_path, "r") as f:
            base_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Base config file not found at {base_config_path}")
        return

    # Create a unique group name for this set of runs
    sweep_group_name = f"Sweep_Loss_Weights_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    base_cfg["sweep_group"] = sweep_group_name

    # --- Generate and Run Experiments ---
    param_keys = list(param_grid.keys())
    value_lists = [param_grid[key] for key in param_keys]
    all_combinations = list(itertools.product(*value_lists))
    total_runs = len(all_combinations)

    print(f"Starting sweep '{sweep_group_name}' with {total_runs} total runs.")
    if not wandb_enabled:
        print("\n*** WANDB LOGGING IS DISABLED ***\n")

    for i, combo in enumerate(all_combinations):
        run_cfg = copy.deepcopy(base_cfg)

        # --- Modify Config for this Run ---
        
        # Pass the master W&B switch to the config
        run_cfg['wandb_enabled'] = wandb_enabled

        run_name_parts = []
        config_summary = []

        for j, key_str in enumerate(param_keys):
            value = combo[j]
            set_nested_key(run_cfg, key_str, value)
            short_name = (
                key_str.split(".")[-1]
                .replace("_weight", "")
                .replace("log2_hashmap_size", "hash")
            )
            run_name_parts.append(f"{short_name}-{value}")
            config_summary.append(f"{key_str}={value}")

        run_cfg["exp_name"] = "_".join(run_name_parts)

        # --- Run Training ---
        try:
            print("\n" + "=" * 50)
            print(f"Starting run ({i+1}/{total_runs}): {run_cfg['exp_name']}")
            print(f"Config: {', '.join(config_summary)}")
            print("=" * 50 + "\n")

            train_main(run_cfg)

            print(f"\nSuccessfully finished run: {run_cfg['exp_name']}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\n" + "!" * 50)
                print(f"FAILED run: {run_cfg['exp_name']} due to CUDA Out of Memory.")
                print("Skipping this combination.")
                print("!" * 50 + "\n")
                if wandb_enabled:
                    wandb.finish(exit_code=1) # Finish the crashed run
            else:
                print(
                    f"\nFAILED run: {run_cfg['exp_name']} with an unexpected error: {e}"
                )
                if wandb_enabled:
                    wandb.finish(exit_code=1) # Finish the crashed run
        except Exception as e:
            print(f"\nFAILED run: {run_cfg['exp_name']} with an unexpected error: {e}")
            if wandb_enabled:
                wandb.finish(exit_code=1) # Finish the crashed run
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # --- Add argument parsing ---
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep.")
    # This setup enables W&B by default.
    # Running with '--no-wandb' will set 'wandb_enabled' to False.
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="wandb_enabled",
        help="Disable Weights & Biases logging for this sweep.",
    )
    args = parser.parse_args()

    run_sweep(args.wandb_enabled)

