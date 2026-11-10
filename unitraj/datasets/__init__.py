def build_dataset(config, val=False):
    if val and config.get("h5_dir"):
        from .h5_dataset import NativeH5Dataset
        return NativeH5Dataset(config.h5_dir)
    from .fmae_dataset import FMAEDataset
    return FMAEDataset(config=config, is_validation=val)
