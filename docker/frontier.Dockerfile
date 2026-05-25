FROM python:3.11-slim

WORKDIR /app
COPY apps/frontier-assistant/ ./apps/frontier-assistant/
COPY core/ ./core/
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir -e .

EXPOSE 7861
CMD ["python", "apps/frontier-assistant/app.py", "--server_port", "7861"]
