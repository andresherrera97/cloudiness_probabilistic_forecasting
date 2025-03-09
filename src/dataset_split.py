import os


def main(train_pct: float = 0.7, val_pct: float = 0.1, test_pct: float = 0.3):
    assert train_pct + val_pct + test_pct == 1.0

    DATASET_PATH = "datasets/salto1024_all/"
    days_folders = sorted(os.listdir(DATASET_PATH))
    num_days = len(days_folders)
    train_end = int(num_days * train_pct)
    val_start, val_end = train_end, train_end + int(num_days * val_pct)
    test_start = val_end

    train_days = days_folders[:train_end]
    val_days = days_folders[val_start:val_end]
    test_days = days_folders[test_start:]

    train_path = os.path.join(DATASET_PATH, "train")
    val_path = os.path.join(DATASET_PATH, "val")
    test_path = os.path.join(DATASET_PATH, "test")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for day in train_days:
        os.rename(os.path.join(DATASET_PATH, day), os.path.join(train_path, day))

    for day in val_days:
        os.rename(os.path.join(DATASET_PATH, day), os.path.join(val_path, day))

    for day in test_days:
        os.rename(os.path.join(DATASET_PATH, day), os.path.join(test_path, day))


if __name__ == "__main__":
    main()
