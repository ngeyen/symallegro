import os
import yaml
import torch
from model import AllegroModel
from nequip.data.datamodule import sGDML_CCSD_DataModule
from nequip.train import EMALightningModule
from lightning import Trainer

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_data(config):
    data_module = sGDML_CCSD_DataModule(
        dataset=config['data']['dataset'],
        data_source_dir=config['data']['data_source_dir'],
        transforms=config['data']['transforms'],
        trainval_test_subset=config['data']['trainval_test_subset'],
        train_val_split=config['data']['train_val_split'],
        seed=config['data']['seed'],
        stats_manager=config['data']['stats_manager']
    )
    return data_module

def setup_model(config):
    model = AllegroModel(
        seed=config['training_module']['model']['seed'],
        model_dtype=config['training_module']['model']['model_dtype'],
        type_names=config['training_module']['model']['type_names'],
        r_max=config['training_module']['model']['r_max'],
        radial_chemical_embed=config['training_module']['model']['radial_chemical_embed'],
        radial_chemical_embed_dim=config['training_module']['model']['radial_chemical_embed_dim'],
        scalar_embed_mlp_hidden_layers_depth=config['training_module']['model']['scalar_embed_mlp_hidden_layers_depth'],
        scalar_embed_mlp_hidden_layers_width=config['training_module']['model']['scalar_embed_mlp_hidden_layers_width'],
        scalar_embed_mlp_nonlinearity=config['training_module']['model']['scalar_embed_mlp_nonlinearity'],
        l_max=config['training_module']['model']['l_max'],
        num_layers=config['training_module']['model']['num_layers'],
        num_scalar_features=config['training_module']['model']['num_scalar_features'],
        num_tensor_features=config['training_module']['model']['num_tensor_features'],
        allegro_mlp_hidden_layers_depth=config['training_module']['model']['allegro_mlp_hidden_layers_depth'],
        allegro_mlp_hidden_layers_width=config['training_module']['model']['allegro_mlp_hidden_layers_width'],
        allegro_mlp_nonlinearity=config['training_module']['model']['allegro_mlp_nonlinearity'],
        parity=config['training_module']['model']['parity'],
        tp_path_channel_coupling=config['training_module']['model']['tp_path_channel_coupling'],
        readout_mlp_hidden_layers_depth=config['training_module']['model']['readout_mlp_hidden_layers_depth'],
        readout_mlp_hidden_layers_width=config['training_module']['model']['readout_mlp_hidden_layers_width'],
        readout_mlp_nonlinearity=config['training_module']['model']['readout_mlp_nonlinearity'],
        avg_num_neighbors=config['training_module']['model']['avg_num_neighbors'],
        per_type_energy_shifts=config['training_module']['model']['per_type_energy_shifts'],
        per_type_energy_scales=config['training_module']['model']['per_type_energy_scales'],
        per_type_energy_scales_trainable=config['training_module']['model']['per_type_energy_scales_trainable'],
        per_type_energy_shifts_trainable=config['training_module']['model']['per_type_energy_shifts_trainable'],
        pair_potential=config['training_module']['model']['pair_potential']
    )
    return model

def main():
    config_path = os.path.join(os.path.dirname(__file__), '../configs/tutorial.yaml')
    config = load_config(config_path)

    data_module = setup_data(config)
    model = setup_model(config)

    trainer = Trainer(
        max_epochs=config['trainer']['max_epochs'],
        check_val_every_n_epoch=config['trainer']['check_val_every_n_epoch'],
        log_every_n_steps=config['trainer']['log_every_n_steps'],
        callbacks=config['trainer']['callbacks']
    )

    training_module = EMALightningModule(
        model=model,
        loss=config['training_module']['loss'],
        val_metrics=config['training_module']['val_metrics'],
        test_metrics=config['training_module']['test_metrics'],
        optimizer=config['training_module']['optimizer']
    )

    trainer.fit(training_module, data_module)

if __name__ == "__main__":
    main()