from src.old.split_reco_dataset_v2 import split_labels


def test_split_is_deterministic_and_covers_all_labels():
    labels = {f"crop_{i}.png": f"word{i}" for i in range(20)}
    train_a, val_a = split_labels(labels, val_fraction=0.2, seed=7)
    train_b, val_b = split_labels(labels, val_fraction=0.2, seed=7)
    assert train_a == train_b and val_a == val_b
    assert set(train_a) | set(val_a) == set(labels)
    assert set(train_a) & set(val_a) == set()


def test_val_fraction_is_approximately_respected():
    labels = {f"crop_{i}.png": f"word{i}" for i in range(20)}
    _, val = split_labels(labels, val_fraction=0.2, seed=7)
    assert len(val) == 4
