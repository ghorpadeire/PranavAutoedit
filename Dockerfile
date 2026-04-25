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

# Health check for Railway/Render uptime monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import config, ai_review, pipeline; print('ok')" || exit 1

# Default: run tests (override with CMD in docker-compose or Railway)
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
