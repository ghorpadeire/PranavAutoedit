# ⚠️  MNC deployment: use Podman instead of Docker Desktop.
# Docker Desktop requires a paid licence for companies >250 employees.
# All commands are identical — just replace 'docker' with 'podman'.
# Install Podman: https://podman.io/docs/installation

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# Health check — hits the /health endpoint once the server is up
# CI overrides CMD to run tests: docker run pranavautoedit python -m unittest discover -s tests -v
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
