FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
SHELL ["/bin/bash", "-c"]
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip compile pyproject.toml -q -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 源码
COPY . .

# 数据目录（通过 volume 挂载，镜像中不包含数据）
RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/data/bm25_index

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
