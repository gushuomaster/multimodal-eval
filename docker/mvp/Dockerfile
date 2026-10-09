FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /opt/evalscope
COPY . .
RUN python -m pip install --no-cache-dir .

WORKDIR /eval/data
ENTRYPOINT ["evalscope"]
CMD ["--help"]
