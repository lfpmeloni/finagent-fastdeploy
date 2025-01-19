FROM python:3.11-slim AS frontend
WORKDIR /frontend
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 3000
CMD ["uvicorn", "frontend_server:app", "--host", "0.0.0.0", "--port", "3000"]