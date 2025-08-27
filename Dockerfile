FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "start_server.py", "--model", "/app/models/qwen7b", "--host", "0.0.0.0", "--port", "8000"]