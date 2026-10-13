FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY app ./app
COPY data/target_glosses.txt ./data/target_glosses.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[app]"

COPY models ./models
RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app/reports/agent_traces \
    && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true"]
