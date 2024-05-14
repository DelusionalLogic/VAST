from torch.utils.data import (
    DataLoader,
)

from .logger import (
    LOGGER,
)


def create_val_dataloaders(_args):

    LOGGER.info(f"Create Dataset {dataset.dataset_name} Success")
    return DataLoader(
        AnnoIndexedDataset(),
        batch_size = 8,
        collate_fn = collate,
    )

