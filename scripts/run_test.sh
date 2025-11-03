#!/bin/bash

# Activate the virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$(pwd)/src

# Run the test script
python -m unittest discover -s tests -p "*.py"