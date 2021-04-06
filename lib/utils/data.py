import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, random_split

from config import BATCH_SIZE, MANUAL_SEED, MAX_TEST_SIZE, MAX_VALIDATION_SIZE
from lib.collator import Collator


def create_data_samples(dataset):
    test_size = min(int(0.2 * len(dataset)), MAX_TEST_SIZE)
    validation_size = min(int(0.2 * len(dataset)), MAX_VALIDATION_SIZE)
    train_size = len(dataset) - test_size - validation_size

    train_set, test_set, validation_set = random_split(
        dataset,
        lengths=[train_size, test_size, validation_size],
        generator=torch.Generator().manual_seed(MANUAL_SEED),
    )

    return train_set, test_set, validation_set


def create_dataloaders(
    dataset,
    train_set,
    validation_set,
    test_set,
    device=torch.device("cpu"),
    modify_train=False,
    evaluate=False,
):
    # item[2] is item index
    validation_set_preference_change_mask = [
        idx
        for idx, item in enumerate(validation_set)
        if item[2] in dataset.modified_session_ids
    ]
    validation_set_preference_change = Subset(
        validation_set, validation_set_preference_change_mask
    )

    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=Collator(device, modify_train),
        ),
        "validation": DataLoader(
            validation_set,
            batch_size=BATCH_SIZE,
            collate_fn=Collator(device, True),
        ),
    }

    if evaluate:
        test_set_preference_change_mask = [
            idx
            for idx, item in enumerate(test_set)
            if item[2] in dataset.modified_session_ids
        ]
        test_set_preference_change = Subset(test_set, test_set_preference_change_mask)

        test_dataloaders = {
            "validation_preference_change_original": DataLoader(
                validation_set_preference_change,
                batch_size=BATCH_SIZE,
                collate_fn=Collator(device, False),
            ),
            "validation_preference_change_modified": DataLoader(
                validation_set_preference_change,
                batch_size=BATCH_SIZE,
                collate_fn=Collator(device, True),
            ),
            "test": DataLoader(
                test_set,
                batch_size=BATCH_SIZE,
                collate_fn=Collator(device, True),
            ),
            "test_preference_change_original": DataLoader(
                test_set_preference_change,
                batch_size=BATCH_SIZE,
                collate_fn=Collator(device, False),
            ),
            "test_preference_change_modified": DataLoader(
                test_set_preference_change,
                batch_size=BATCH_SIZE,
                collate_fn=Collator(device, True),
            ),
        }

        dataloaders.update(test_dataloaders)

    return dataloaders
