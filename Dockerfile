FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "crypto_bot.main"]
