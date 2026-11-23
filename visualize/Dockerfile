# Lekki obraz bazowy
FROM python:3.11-slim

# Katalog roboczy
WORKDIR /app

# Kod serwera wynikow
COPY serve.py .

# Port serwera HTTP
EXPOSE 8080

# Domyslne uruchomienie
CMD ["python", "-u", "serve.py"]
