# Original prediction checkpoints

- `mdp.ckpt`: original epoch-26 MDP Lightning checkpoint selected at val/minFDE6=2.0812.
- `forecast_mae.ckpt`: original epoch-23 Forecast-MAE Lightning checkpoint selected at val/brier_fde=2.89.

Both files are byte-identical copies of the original checkpoints; only their filenames are shortened. They have not been converted or stripped. Testing reads state_dict and removes the trainer's net. prefix in memory. All original model state entries are loaded strictly; the only entries skipped are the unused `linear_cli` / `linear_modal` projection heads and the `modal_*` contrastive queues, which no longer exist in the network (see `unitraj/models/checkpoint.py`). Every other head, loss term and queue is still loaded and checked.

The default evaluator and paired trainers both load these original .ckpt files. No AE weights or separate MAE pretraining weights are provided or required.

Original Lightning checkpoints contain pickled experiment metadata, optimizer states and potentially local paths. Load only trusted files; inspect their metadata separately before anonymous public redistribution. For this release the embedded absolute paths and account names were replaced in place by neutral placeholders of identical length (`/local/anonusr/anonuser/...`), so the pickle stream and every tensor are unchanged.
