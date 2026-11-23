from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    """
    Wczytuje dane z pliku CSV i zwraca DataFrame.

    Parametry:
        path: sciezka do pliku CSV

    Zwraca:
        pandas.DataFrame
    """
    path = Path(path)

    # Kontrola istnienia pliku
    if not path.is_file():
        raise FileNotFoundError(
            f"Nie znaleziono pliku danych: {path}"
        )

    # Wczytanie danych
    df = pd.read_csv(path)

    # Weryfikacja niepustego zbioru
    if df.empty:
        raise ValueError("Wczytany zbior danych jest pusty.")

    # Kontrola obecnosci etykiety klasy
    if "target" not in df.columns:
        raise ValueError(
            "Brak wymaganej kolumny 'target'."
        )

    return df


def split_data(df, test_size, seed):
    """
    Dzieli dane na zbior treningowy i testowy.

    Zwraca:
        X_train, X_test, y_train, y_test
    """

    # Kontrola poprawnosci danych
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dane musza byc typu DataFrame.")

    if df.empty:
        raise ValueError("Nie mozna podzielic pustego zbioru.")

    if "target" not in df.columns:
        raise ValueError("Brak kolumny 'target'.")

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size musi byc wieksze od 0 i mniejsze od 1."
        )

    # Oddzielenie cech i etykiet
    X = df.drop(columns=["target"])
    y = df["target"]

    # Podzial z zachowaniem proporcji klas
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
