import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
from config import get_config
from data_prep import load_data, split_data


def evaluate_model():
    """
    Wczytuje zapisany model, ocenia go
    i zapisuje metryki do pliku tekstowego.
    """
    config = get_config()

    print("EWALUACJA MODELU")

    # Wczytanie danych
    df = load_data(config["DATA_PATH"])

    # Odtworzenie identycznego podzialu
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config["TEST_SIZE"],
        seed=config["RANDOM_SEED"]
    )

    # Lokalizacja wytrenowanego modelu
    model_path = config["MODEL_PATH"]

    # Kontrola istnienia modelu
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {model_path}. "
            "Najpierw uruchom train.py."
        )

    # Wczytanie zapisanego Pipeline
    model = joblib.load(model_path)

    # Predykcja bez ponownego treningu
    y_pred = model.predict(X_test)

    # Obliczenie metryk
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    # Przygotowanie pliku wynikowego
    results_path = config["RESULTS_PATH"]
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Zapis wynikow
    with open(results_path, "w", encoding="utf-8") as file:
        file.write("WYNIKI EWALUACJI MODELU ML\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Zbior danych: {config['DATA_PATH']}\n")
        file.write(f"Model: {config['MODEL_TYPE']}\n")
        file.write(f"Test size: {config['TEST_SIZE']}\n")
        file.write(f"Random seed: {config['RANDOM_SEED']}\n")
        file.write(f"Liczba probek testowych: {len(y_test)}\n")
        file.write(f"Dokladnosc: {accuracy:.4f}\n\n")
        file.write("CLASSIFICATION REPORT\n")
        file.write(report)

    print(f"\nWyniki zapisano w: {results_path}")

    return accuracy


if __name__ == "__main__":
    evaluate_model()
