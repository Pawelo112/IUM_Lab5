import time
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config import get_config
from data_prep import load_data, split_data


def train_model():
    """
    Wczytuje dane, trenuje model i zapisuje go do pliku.
    """
    config = get_config()

    print("TRENING MODELU")

    # Wczytanie danych
    df = load_data(config["DATA_PATH"])

    # Podzial danych
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config["TEST_SIZE"],
        seed=config["RANDOM_SEED"]
    )

    print(f"Liczba obserwacji: {len(df)}")
    print(f"Zbior treningowy: {X_train.shape}")
    print(f"Zbior testowy: {X_test.shape}")

    # Utworzenie Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=config["MAX_ITER"],
            random_state=config["RANDOM_SEED"]
        ))
    ])

    # Pomiar czasu treningu
    start = time.perf_counter()

    # Trening klasyfikatora
    model.fit(X_train, y_train)

    train_time = time.perf_counter() - start

    # Utworzenie katalogu modeli
    model_path = config["MODEL_PATH"]
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Zapis wytrenowanego modelu
    joblib.dump(model, model_path)

    print(f"Czas treningu: {train_time:.4f} s")
    print(f"Model zapisano: {model_path}")
    print("Trening zakonczony poprawnie.")

    return model


if __name__ == "__main__":
    train_model()
