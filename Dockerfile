# Obraz bazowy Python 3.11
FROM python:3.11-slim

# Katalog roboczy
WORKDIR /app

# Skopiowanie listy zaleznosci
COPY requirements.txt .

# Instalacja bibliotek ML
RUN python -m pip install --no-cache-dir -r requirements.txt

# Skopiowanie modulow z laboratorium 3
COPY config.py data_prep.py train.py evaluate.py ./

# Skopiowanie szablonu konfiguracji
COPY .env.example ./

# Skrypt sterujacy calym pipeline'em
COPY runner.py ./

# Port wewnetrznej uslugi statusowej
EXPOSE 5000

# Domyslny trening modelu
CMD ["python", "-u", "train.py"]
