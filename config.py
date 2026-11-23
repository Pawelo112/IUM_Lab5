import os
from pathlib import Path

from dotenv import load_dotenv


# Katalog glowny projektu
BASE_DIR = Path(__file__).resolve().parent

# Wczytanie konfiguracji z pliku .env
load_dotenv(BASE_DIR / ".env")


def get_required_env(name):
    """
    Pobiera wymagana zmienna srodowiskowa.
    Zglasza blad, jesli zmienna nie istnieje lub jest pusta.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        raise ValueError(
            f"Brak wymaganego parametru w .env: {name}"
        )

    return value.strip()


def get_config():
    """
    Zwraca konfiguracje projektu.
    Sciezki sa wyznaczane wzgledem katalogu projektu.
    """
    config = {
        "DATA_PATH": BASE_DIR / get_required_env("DATA_PATH"),
        "MODEL_PATH": BASE_DIR / get_required_env("MODEL_PATH"),
        "RESULTS_PATH": BASE_DIR / get_required_env("RESULTS_PATH"),

        "TEST_SIZE": float(get_required_env("TEST_SIZE")),
        "RANDOM_SEED": int(get_required_env("RANDOM_SEED")),

        "MODEL_TYPE": get_required_env("MODEL_TYPE"),
        "MAX_ITER": int(get_required_env("MAX_ITER")),
    }

    # Walidacja parametrow
    if not 0 < config["TEST_SIZE"] < 1:
        raise ValueError("TEST_SIZE musi nalezec do (0, 1).")

    if config["MAX_ITER"] <= 0:
        raise ValueError("MAX_ITER musi byc dodatni.")

    if config["MODEL_TYPE"] != "logistic_regression":
        raise ValueError(
            "Nieobslugiwany MODEL_TYPE. "
            "Dostepny: logistic_regression."
        )

    return config
