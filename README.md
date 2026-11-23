# IUM Lab 5/6 - Machine Learning CI Pipeline

![ML CI Pipeline](https://github.com/Pawelo112/IUM_Lab5/actions/workflows/ci.yml/badge.svg?branch=main)

## Opis projektu

Projekt przedstawia modularny pipeline uczenia maszynowego
oparty na zbiorze Wine oraz klasyfikatorze LogisticRegression.

Aplikacja wykorzystuje Docker i Docker Compose do uruchamiania
treningu oraz udostępniania wyników.

W laboratorium 6 dodano automatyczny pipeline CI
z wykorzystaniem GitHub Actions.

## Wykorzystane technologie

- Python 3.11
- scikit-learn
- pandas i NumPy
- pytest
- Docker i Docker Compose
- GitHub Actions
- GitHub Container Registry

## Struktura projektu

- `data_prep.py` - wczytywanie i podział danych
- `train.py` - trenowanie i zapis modelu
- `evaluate.py` - ewaluacja modelu
- `runner.py` - uruchamianie pipeline'u z labu 5
- `tests/` - testy jednostkowe
- `Dockerfile` - obraz środowiska ML
- `docker-compose.yml` - konfiguracja usług
- `.github/workflows/ci.yml` - automatyczna integracja CI

## GitHub Actions

Workflow uruchamia się automatycznie po pushu do gałęzi
`main` oraz przy Pull Requestach do tej gałęzi.

Etapy:

1. Budowanie obrazu ML i cache zależności.
2. Uruchamianie testów pytest w kontenerze Docker.
3. Weryfikacja obecności sekretu.
4. Trening klasyfikatora.
5. Ewaluacja modelu.
6. Zapis modelu i raportu jako artefaktów.

## Wyniki

Zbiór danych: Wine.

Model: LogisticRegression ze standaryzacją StandardScaler.

Podział danych: 80% trening, 20% test.

Dokładność uzyskana w laboratorium 5: **97,22%**.

## Uruchomienie lokalne

Z użyciem Docker Compose:

```powershell
docker compose up -d --build
```

Raport wyników jest dostępny pod adresem:

http://localhost:8085/wyniki.txt

Zatrzymanie środowiska:

```powershell
docker compose down
```

## Artefakty CI

Po poprawnym przebiegu GitHub Actions można pobrać:

- `wine-model` - wytrenowany model `.joblib`
- `wine-results` - raport ewaluacji `.txt`

Artefakty są dostępne w szczegółach przebiegu w zakładce Actions.
