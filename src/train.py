import os
from omegaconf import OmegaConf

from nequip.data.datamodule import sGDML_CCSD_DataModule
from nequip.train import EMALightningModule
from lightning import Trainer
from model.allegro_models import AllegroModel
from nequip.utils.global_state import set_global_state, global_state_initialized



def init_nequip_global_state(config):
    allow_tf32 = bool(config.get("global_options", {}).get("allow_tf32", False))
    set_global_state(allow_tf32=allow_tf32)


def _resolve_data_source_dir(raw_dir: str) -> str:
    """Return an absolute existing path for the data_source_dir."""
    if os.path.isabs(raw_dir):
        return raw_dir
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, raw_dir),
        os.path.join(root, "data", raw_dir),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not locate data_source_dir '{raw_dir}'. Tried: {candidates}"
    )


from hydra.utils import instantiate

def setup_data(config):
    return instantiate(config.data)


def _inject_training_stats_into_cfg(cfg, data_module):
    '''Extract training data statistics from data module and inject into config.'''
    
    data_module.setup(stage="fit")
    stats = None
    
    if hasattr(data_module, "stats_manager"):
        sm = data_module.stats_manager
        # Try method to get full stats object
        for meth in ("compute_stats", "get_stats"):
            if hasattr(sm, meth):
                try:
                    stats = getattr(sm, meth)()
                    break
                except Exception:
                    pass
        # Fallback: collect attributes
        if stats is None:
            stats = {}
            for k in ("num_neighbors_mean", "per_atom_energy_mean", "forces_rms"):
                if hasattr(sm, k):
                    stats[k] = getattr(sm, k)

    # Fallback defaults
    if not stats:
        type_names = list(cfg.get("model_type_names", [])) or ["X"]
        stats = {
            "num_neighbors_mean": 12.0,
            "per_atom_energy_mean": [0.0] * len(type_names),
            "forces_rms": [1.0] * len(type_names),
        }

    # Ensure per-type lists become dicts keyed by symbols
    type_names = list(cfg.get("model_type_names", [])) or list(cfg.training_module.model.type_names)
    def list_to_dict(val):
        if isinstance(val, (list, tuple)) and len(val) == len(type_names):
            return {sym: float(v) for sym, v in zip(type_names, val)}
        return val
    stats["per_atom_energy_mean"] = list_to_dict(stats.get("per_atom_energy_mean"))
    stats["forces_rms"] = list_to_dict(stats.get("forces_rms"))

    # Convert to plain python types
    def _to_plain(v):
        try:
            import torch, numpy as np
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            if isinstance(v, np.ndarray):
                return v.tolist()
        except Exception:
            pass
        return v

    stats_plain = {k: _to_plain(v) for k, v in stats.items()}
    # Inject
    cfg.training_data_stats = OmegaConf.create(stats_plain)


def setup_model(config):
    if not global_state_initialized():
        init_nequip_global_state(config)

    # Ensure training_data_stats present
    if "training_data_stats" not in config:
        # Build data module to get stats
        _inject_training_stats_into_cfg(config, setup_data(config))

    model_cfg = OmegaConf.to_container(config.training_module.model, resolve=True)

    model = AllegroModel(
        seed=model_cfg["seed"],
        model_dtype=model_cfg["model_dtype"],
        type_names=model_cfg["type_names"],
        r_max=model_cfg["r_max"],
        radial_chemical_embed=model_cfg["radial_chemical_embed"],
        radial_chemical_embed_dim=model_cfg["radial_chemical_embed_dim"],
        scalar_embed_mlp_hidden_layers_depth=model_cfg["scalar_embed_mlp_hidden_layers_depth"],
        scalar_embed_mlp_hidden_layers_width=model_cfg["scalar_embed_mlp_hidden_layers_width"],
        scalar_embed_mlp_nonlinearity=model_cfg["scalar_embed_mlp_nonlinearity"],
        l_max=model_cfg["l_max"],
        num_layers=model_cfg["num_layers"],
        num_scalar_features=model_cfg["num_scalar_features"],
        num_tensor_features=model_cfg["num_tensor_features"],
        allegro_mlp_hidden_layers_depth=model_cfg["allegro_mlp_hidden_layers_depth"],
        allegro_mlp_hidden_layers_width=model_cfg["allegro_mlp_hidden_layers_width"],
        allegro_mlp_nonlinearity=model_cfg["allegro_mlp_nonlinearity"],
        parity=model_cfg["parity"],
        tp_path_channel_coupling=model_cfg["tp_path_channel_coupling"],
        readout_mlp_hidden_layers_depth=model_cfg["readout_mlp_hidden_layers_depth"],
        readout_mlp_hidden_layers_width=model_cfg["readout_mlp_hidden_layers_width"],
        readout_mlp_nonlinearity=model_cfg["readout_mlp_nonlinearity"],
        avg_num_neighbors=model_cfg["avg_num_neighbors"],
        per_type_energy_shifts=model_cfg["per_type_energy_shifts"],
        per_type_energy_scales=model_cfg["per_type_energy_scales"],
        per_type_energy_scales_trainable=model_cfg["per_type_energy_scales_trainable"],
        per_type_energy_shifts_trainable=model_cfg["per_type_energy_shifts_trainable"],
        pair_potential=model_cfg["pair_potential"],
    )
    return model


def _infer_num_datasets(dm):
    def _count(ds):
        if ds is None:
            return 0
        if isinstance(ds, (list, tuple)):
            return len(ds)
        return 1
    return {
        "train": _count(getattr(dm, "train_dataset", None)),
        "val": _count(getattr(dm, "val_dataset", None)),
        "test": _count(getattr(dm, "test_dataset", None)),
        "predict": 0,
    }

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../configs/tutorial.yaml")
    config = OmegaConf.load(config_path)

    init_nequip_global_state(config)

    data_module = setup_data(config)
    # ensure datasets are built
    try:
        data_module.setup(stage="fit")
    except TypeError:
        data_module.setup()

    # inject stats 
    _inject_training_stats_into_cfg(config, data_module)

    num_datasets = _infer_num_datasets(data_module)
    if num_datasets["train"] != 1:
        raise RuntimeError(f"Expected exactly 1 train dataset, found {num_datasets['train']}.")

    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)
    tm_cfg = OmegaConf.to_container(config.training_module, resolve=True)
    model_cfg = OmegaConf.to_container(config.training_module.model, resolve=True)

    # WandB Logger
    from lightning.pytorch.loggers import WandbLogger
    wandb_logger = WandbLogger(project="allegro-training", log_model=True)

    trainer = Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 1),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        logger=wandb_logger,
    )

    training_module = EMALightningModule(
        model=model_cfg,
        loss=tm_cfg["loss"],
        val_metrics=tm_cfg["val_metrics"],
        test_metrics=tm_cfg["test_metrics"],
        optimizer=tm_cfg["optimizer"],
        num_datasets=num_datasets,
    )

    trainer.fit(training_module, data_module)

if __name__ == "__main__":
    main()