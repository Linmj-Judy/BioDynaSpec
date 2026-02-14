import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset

from .dataset_atlas import (
    ProteinMotionDataset,
    ProteinMotionWindowDataset,
    ProteinMotionFFTWindowDataset,
    collate_fn,
    collate_fn_single,
)

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def setup(self, stage=None):
        # Load training dataset
        if "stft" in self.config.train_path:
            self.train_dataset = ProteinMotionWindowDataset(
                data_path=self.config.train_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
        elif "fft_window" in self.config.train_path:
            self.train_dataset = ProteinMotionFFTWindowDataset(
                data_path=self.config.train_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
        else:
            self.train_dataset = ProteinMotionDataset(
                data_path=self.config.train_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
            
        # Load validation dataset
        if "stft" in self.config.val_path:
            self.val_dataset = ProteinMotionWindowDataset(
                data_path=self.config.val_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
        elif "fft_window" in self.config.val_path:
            self.val_dataset = ProteinMotionFFTWindowDataset(
                data_path=self.config.val_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
        else:
            self.val_dataset = ProteinMotionDataset(
                data_path=self.config.val_path,
                bihito_repr_dir=self.config.bihito_repr_dir,
            )
        
        print(f"\nDataset Setup:")
        print(f"Training set size: {len(self.train_dataset)}")
        print(f"Validation set size: {len(self.val_dataset)}")
    
    def train_dataloader(self):
        is_iterable = isinstance(self.train_dataset, IterableDataset)
        
        train_ds = self.train_dataset
        
        # 动态子采样逻辑
        if (not is_iterable) and hasattr(self.config, 'use_dataset_subsample') and self.config.use_dataset_subsample:
            ratio = getattr(self.config, 'subsample_ratio', 0.1)
            total_len = len(self.train_dataset)
            subset_len = max(1, int(total_len * ratio))
            
            # 使用 random_split 随机采样，每次调用 train_dataloader 都会重新采样（如果是在每个 epoch 重新创建 loader 的情况下）
            # 注意：PyTorch Lightning 默认会在每个 epoch 重新调用 train_dataloader (如果 reload_dataloaders_every_n_epochs > 0)
            # 或者你可以手动实现一个 RandomSampler 更加高效
            
            from torch.utils.data import RandomSampler
            sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=subset_len)
            shuffle = False # Sampler 和 shuffle 不能共存
        else:
            sampler = None
            shuffle = None if is_iterable else True

        return DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_single,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None
        )
    
    def val_dataloader(self):
        is_iterable = isinstance(self.val_dataset, IterableDataset)
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=None if is_iterable else False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn_single,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=None
        )
