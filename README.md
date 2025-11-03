# Allegro Local Development

## Overview
This project is a local development environment for the Allegro model, designed for research and experimentation. The Allegro model is a machine learning framework for simulating molecular interactions and properties.

## Project Structure
- **src/allegro**: Contains the source code for the Allegro model, including modules and classes that define the model's architecture and functionality.
- **src/train.py**: The main script for training the Allegro model. It sets up the training process, loads data, and initiates the training loop.
- **src/__init__.py**: Marks the directory as a Python package and may include initialization code.
- **configs/tutorial.yaml**: Configuration settings for running the Allegro model, specifying parameters for training, data loading, and model architecture.
- **scripts/run_train.sh**: Shell script to execute the training process, setting up the environment and calling the training script.
- **scripts/run_test.sh**: Shell script to execute the testing process, setting up the environment and calling the testing script.
- **data/aspirin_data**: Contains the dataset used for training and testing the Allegro model.
- **tests/test_imports.py**: Test cases to verify that the Allegro model and all its component are working.
- **pyproject.toml**: Configuration file specifying dependencies, build system requirements, and project metadata.
- **requirements.txt**: Lists the Python packages required for the project.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **.env.example**: Provides an example of environment variables for local development.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd allegro
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy `.env.example` to `.env` and modify it as needed.

## Usage
- To train the model, run:
  ```
  ./scripts/run_train.sh
  ```

- To test the model, run:
  ```
  ./scripts/run_test.sh
  ```


## License
This project is licensed under the MIT License. See the LICENSE file for details.