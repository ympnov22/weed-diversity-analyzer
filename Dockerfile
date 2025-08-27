# Lightweight Alpine-based build for minimal memory footprint
FROM python:3.12-alpine as builder

# Set environment variables for build stage
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies including C++ compiler for scikit-learn
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    postgresql-dev \
    gfortran \
    openblas-dev \
    lapack-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy minimal requirements and install Python dependencies
COPY requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Production stage
FROM python:3.12-alpine as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies including OpenMP and C++ stdlib for scikit-learn
RUN apk add --no-cache \
    postgresql-libs \
    curl \
    libgomp \
    libstdc++

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app directory and user
WORKDIR /app
RUN adduser -D -s /bin/sh appuser

# Copy only essential application files
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Create minimal necessary directories
RUN mkdir -p logs temp \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
