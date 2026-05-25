FROM python:3.11-slim

WORKDIR /app
COPY apps/oss-assistant/ ./apps/oss-assistant/
COPY core/ ./core/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .

EXPOSE 7860
CMD ["python", "apps/oss-assistant/app.py", "--server_port", "7860"]
