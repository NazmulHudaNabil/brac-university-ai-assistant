# Use official Python lightweight image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PORT=8000

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY data/golden/ ./data/golden/

# Create sessions directory and change ownership for non-root execution
RUN mkdir -p .sessions && \
    useradd -m -r appuser && \
    chown -R appuser:appuser /workspace

# Run as non-root user for security (ECS Fargate best practice)
USER appuser

# Expose port for FastAPI (8000)
EXPOSE 8000

# Default command (FastAPI Backend)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
