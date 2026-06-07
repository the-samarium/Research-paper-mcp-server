FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

RUN mkdir -p /app/chromaDB /app/assets

EXPOSE 8000

CMD ["uv", "run", "fastmcp", "run", "server.py", "--port", "8000", "--host", "0.0.0.0"]