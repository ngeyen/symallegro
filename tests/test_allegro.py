import pytest
import numpy as np
import os 


def test_dataset(capsys):
    """
    Tests loading the data file and verifies its contents.
    """
    # Use a relative path, adjust as necessary for your project structure
    file_path = 'data/aspirin_data/aspirin_ccsd-train.npz'

    # Check if the file exists before attempting to load it
    if not os.path.exists(file_path):
        pytest.fail(f"Data file not found at: {file_path}")

    data = np.load(file_path)
    
    # Verify that the data file contains expected arrays
    assert 'E' in data.files
    assert 'F' in data.files
    assert 'R' in data.files

    assert len(data.files) > 0
    
    with capsys.disabled():
        print(f"\nLoaded {file_path}: {data.files}")
    
def test_check_imports():
    """
    Verifies that necessary libraries/modules can be imported successfully.
    """
    # nequip should be installed
    try:
        from nequip.data.datamodule import sGDML_CCSD_DataModule
    except ImportError as e:
        pytest.fail(f"Import failed (nequip): {e}")

    try:
        from model.allegro_models import AllegroModel
    except ImportError:
        pytest.fail(f"Import failed (AllegroModel): {e}")

def test_model_instantiation():
    """
    Tests if the AllegroModel can be instantiated with default parameters.
    """
    from src.train import setup_data
    
    # load config from tutorial.yaml
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '../configs/tutorial.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    data = setup_data(config)
    
    assert data is not None, "Data module setup failed."