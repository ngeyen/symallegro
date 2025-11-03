#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set the configuration file path
CONFIG_FILE="configs/tutorial.yaml"

# Run the training script
python src/train.py --config $CONFIG_FILE "$@"