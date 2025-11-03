import os
from nequip.data.datamodule import sGDML_CCSD_DataModule

root = os.path.dirname(os.path.dirname(__file__))
out_dir = os.path.join(root, "data", "aspirin_data")
os.makedirs(out_dir, exist_ok=True)

dm = sGDML_CCSD_DataModule(
	transforms=[],
	seed=42,
	train_val_split=(0.8, 0.1, 0.1),
	dataset="aspirin",
	data_source_dir=out_dir,
)
dm.prepare_data()
print(f"Downloaded to: {out_dir}")