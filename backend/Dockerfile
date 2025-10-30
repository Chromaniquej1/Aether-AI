# ModelForge-CV Backend Dockerfile
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Base image with CUDA support
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================================================
# Stage 2: Dependencies installation
# ============================================================================
FROM base as dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies as application

WORKDIR /app

# Copy application code
COPY backend/ ./backend/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/workspaces /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Stage 4: Development image (optional)
# ============================================================================
FROM application as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install Jupyter for notebooks
RUN pip install --no-cache-dir jupyter jupyterlab

# Set development environment
ENV ENVIRONMENT=development

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]