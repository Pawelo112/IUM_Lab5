import pandas as pd
import pytest

from data_prep import load_data, split_data


# Przykladowy zbior uzywany w testach
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_1": list(range(20)),
        "feature_2": [x * 2 for x in range(20)],
        "target": [0] * 10 + [1] * 10
    })


def test_split_sizes(sample_df):
    """
    Test 1: poprawna proporcja podzialu 80/20.
    """
    X_train, X_test, y_train, y_test = split_data(
        sample_df,
        test_size=0.2,
        seed=42
    )

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_split_total_size(sample_df):
    """
    Test 2: suma probek po podziale
    musi byc rowna liczbie oryginalnych probek.
    """
    X_train, X_test, y_train, y_test = split_data(
        sample_df,
        test_size=0.2,
        seed=42
    )

    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


def test_split_empty_dataframe():
    """
    Test 3: pusty DataFrame powinien wywolac ValueError.
    """
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        split_data(
            empty_df,
            test_size=0.2,
            seed=42
        )


def test_split_invalid_test_size(sample_df):
    """
    Test 4: niepoprawne test_size powinno
    wywolac ValueError.
    """
    with pytest.raises(ValueError):
        split_data(
            sample_df,
            test_size=1.5,
            seed=42
        )


def test_load_data(tmp_path):
    """
    Test 5: poprawne wczytanie danych z pliku CSV.
    """
    csv_path = tmp_path / "test_data.csv"

    original_df = pd.DataFrame({
        "feature_1": [1, 2, 3],
        "feature_2": [4, 5, 6],
        "target": [0, 1, 0]
    })

    original_df.to_csv(csv_path, index=False)

    loaded_df = load_data(csv_path)

    pd.testing.assert_frame_equal(
        loaded_df,
        original_df
    )


def test_load_missing_file(tmp_path):
    """
    Test 6: brak pliku powinien wywolac
    FileNotFoundError.
    """
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


# Celowo niepoprawny test do sprawdzenia CI
def test_intentional_ci_failure():
    assert 2 + 2 == 5

