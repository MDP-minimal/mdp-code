# Original prediction checkpoints

- `mdp.ckpt`: original epoch-26 MDP Lightning checkpoint selected at val/minFDE6=2.0812.
- `forecast_mae.ckpt`: original epoch-23 Forecast-MAE Lightning checkpoint selected at val/brier_fde=2.89.

Both files are byte-identical copies of the original checkpoints; only their filenames are shortened. They have not been converted or stripped. Testing reads state_dict and removes the trainer's net. prefix in memory. The original MDP contrastive heads and queue buffers are now retained. All original model state entries are loaded strictly; no training heads or queues are filtered out.

The default evaluator and paired trainers both load these original .ckpt files. No AE weights or separate MAE pretraining weights are provided or required.

Original Lightning checkpoints contain pickled experiment metadata, optimizer states and potentially local paths. Load only trusted files; inspect their metadata separately before anonymous public redistribution. For this release the embedded absolute paths and account names were replaced in place by neutral placeholders of identical length (`/local/anonusr/anonuser/...`), so the pickle stream and every tensor are unchanged.
