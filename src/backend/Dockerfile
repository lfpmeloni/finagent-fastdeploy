# Base Python image
FROM python:3.11


# Backend app setup
WORKDIR /app/backend
COPY . .  
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
