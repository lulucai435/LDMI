import os
import argparse
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from main import resolve_env_variables

def main():
    """Loads a trained model and runs the specified experiment."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run model experiment.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument("--experiment", type=str, required=True, help="Path to the experiment configuration file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu).")

    args = parser.parse_args()

    # Load model configuration
    print(f"Loading model config from {args.model_cfg}...")
    model_config = OmegaConf.load(args.model_cfg)
    model_config = resolve_env_variables(model_config)

    # Instantiate model
    print(f"Loading model from {args.ckpt}...")
    model = instantiate_from_config(model_config.model)
    
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()
    print("Model loaded successfully.")

    # Load experiment configuration
    print(f"Loading experiment config from {args.experiment}...")
    exp_cfg = OmegaConf.load(args.experiment)
    experiment = instantiate_from_config(exp_cfg)

    # Load dataset
    print("Loading dataset...")
    data = instantiate_from_config(model_config.data)
    data.prepare_data()
    data.setup()

    # Extract logdir from ckpt file
    logdir = os.path.dirname(os.path.dirname(args.ckpt))

    # Run the experiment
    print("Running experiment...")
    trainer = None  # Assuming trainer is not needed, modify if necessary
    experiment.run(model, data, logdir)

    print("Experiment completed successfully.")

if __name__ == "__main__":
    main()