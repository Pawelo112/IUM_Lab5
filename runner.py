import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from config import get_config
from train import train_model
from evaluate import evaluate_model


# Stan wykonywania pipeline'u
pipeline_status = {
    "status": "starting",
    "model_ready": False,
    "accuracy": None
}


class StatusHandler(BaseHTTPRequestHandler):
    """
    Serwer udostepniajacy status wytrenowanego modelu.
    """

    def do_GET(self):
        if self.path != "/status":
            self.send_error(404, "Nieznana sciezka")
            return

        content = json.dumps(
            pipeline_status,
            indent=2
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header(
            "Content-Length",
            str(len(content))
        )
        self.end_headers()

        self.wfile.write(content)


def main():
    print("=== PIPELINE ML - LABORATORIUM 5 ===", flush=True)

    start_time = time.perf_counter()

    # Etap 1 - trening
    print("\n[1/3] TRENING MODELU", flush=True)
    train_model()

    # Etap 2 - ewaluacja
    print("\n[2/3] EWALUACJA MODELU", flush=True)
    accuracy = evaluate_model()

    # Kontrola zapisanych artefaktow
    config = get_config()

    model_exists = config["MODEL_PATH"].is_file()
    results_exists = config["RESULTS_PATH"].is_file()

    if not model_exists or not results_exists:
        raise RuntimeError(
            "Brak modelu lub raportu po zakonczeniu pipeline'u."
        )

    elapsed = time.perf_counter() - start_time

    pipeline_status["status"] = "completed"
    pipeline_status["model_ready"] = True
    pipeline_status["accuracy"] = round(float(accuracy), 4)

    print("\n[3/3] PIPELINE ZAKONCZONY", flush=True)
    print(f"Accuracy: {accuracy:.4f}", flush=True)
    print(f"Czas pipeline'u: {elapsed:.4f} s", flush=True)
    print(f"Model zapisany: {config['MODEL_PATH']}", flush=True)
    print(f"Wyniki zapisane: {config['RESULTS_PATH']}", flush=True)

    # Serwer statusu utrzymuje kontener aktywny
    server = HTTPServer(
        ("0.0.0.0", 5000),
        StatusHandler
    )

    print(
        "Serwer statusu dziala na porcie 5000",
        flush=True
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
