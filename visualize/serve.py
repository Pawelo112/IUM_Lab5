import json
import os
import time
import urllib.request
from pathlib import Path
from http.server import (
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer
)

# Konfiguracja uslugi
TRAIN_STATUS_URL = os.getenv(
    "TRAIN_STATUS_URL",
    "http://train:5000/status"
)

PORT = int(os.getenv("PORT", "8080"))

RESULTS_DIR = Path("/site")
MODELS_DIR = Path("/models")


def check_train_service():
    """
    Sprawdza dostepnosc uslugi train przez siec Compose.
    """
    print(
        f"Sprawdzanie uslugi: {TRAIN_STATUS_URL}",
        flush=True
    )

    # Kilka prob na wypadek opoznien sieciowych
    for attempt in range(1, 11):
        try:
            with urllib.request.urlopen(
                    TRAIN_STATUS_URL,
                    timeout=5
            ) as response:
                data = json.load(response)

            if data.get("model_ready") is True:
                print(
                    "Polaczenie z usluga train: OK",
                    flush=True
                )
                print(
                    f"Status treningu: {data['status']}",
                    flush=True
                )
                print(
                    f"Accuracy modelu: {data['accuracy']}",
                    flush=True
                )
                return

        except Exception as error:
            print(
                f"Proba {attempt}: {error}",
                flush=True
            )

        time.sleep(2)

    raise RuntimeError(
        "Nie udalo sie potwierdzic gotowosci train."
    )


def check_shared_files():
    """
    Weryfikuje dostep do wspolnych artefaktow.
    """
    results_file = RESULTS_DIR / "wyniki.txt"
    model_file = MODELS_DIR / "wine_model.joblib"

    if not results_file.is_file():
        raise FileNotFoundError(
            f"Brak raportu: {results_file}"
        )

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Brak modelu: {model_file}"
        )

    print(
        f"Raport widoczny: {results_file}",
        flush=True
    )
    print(
        f"Model widoczny: {model_file}",
        flush=True
    )


def main():
    print("=== USLUGA VISUALIZE ===", flush=True)

    # Test polaczenia po nazwie uslugi train
    check_train_service()

    # Sprawdzenie wspolnych wolumenow
    check_shared_files()

    # Udostepnianie katalogu wynikow przez HTTP
    def handler(*args, **kwargs):
        return SimpleHTTPRequestHandler(
            *args,
            directory=str(RESULTS_DIR),
            **kwargs
        )

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        handler
    )

    print(
        f"Serwer wynikow uruchomiony na porcie {PORT}",
        flush=True
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
