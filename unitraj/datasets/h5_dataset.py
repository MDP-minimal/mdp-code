"""Read native UniTraj H5 features. No custom label pipeline."""
from pathlib import Path
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

NATIVE_KEYS = (
    "center_objects_id", "center_objects_world", "obj_trajs_mask", "obj_trajs_pos",
    "obj_trajs", "obj_trajs_future_state", "map_polylines", "map_polylines_mask",
    "obj_trajs_future_mask", "scenario_id", "center_gt_trajs", "center_gt_trajs_mask",
    "center_gt_final_valid_idx",
)

class NativeH5Dataset(Dataset):
    def __init__(self, directory, limit=None):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.entries = []
        self.handles = {}
        index = self.directory / "file_list.pkl"
        if index.is_file():
            with index.open("rb") as stream:
                metadata = pickle.load(stream)
            for key, info in metadata.items():
                file = self.directory / Path(info["h5_path"]).name
                if not file.is_file():
                    raise FileNotFoundError(file)
                self.entries.append((str(file), key))
        else:
            for file in sorted(self.directory.glob("*.h5")):
                with h5py.File(file, "r") as stream:
                    self.entries.extend((str(file), key) for key in stream)
        if limit is not None:
            self.entries = self.entries[:limit]
        if not self.entries:
            raise ValueError("No H5 sample groups found")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        file, key = self.entries[index]
        if file not in self.handles:
            self.handles[file] = h5py.File(file, "r")
        group = self.handles[file][key]
        missing = [key for key in NATIVE_KEYS if key not in group]
        if missing:
            raise KeyError("Missing native UniTraj features: " + str(missing))
        record = {}
        for name in NATIVE_KEYS:
            value = group[name][()]
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            record[name] = value
        return record

    @staticmethod
    def collate_fn(records):
        data = {}
        for key in NATIVE_KEYS:
            values = [record[key] for record in records]
            data[key] = values if isinstance(values[0], str) else torch.from_numpy(np.stack(values))
        return {"input_dict": data, "batch_size": len(records)}
