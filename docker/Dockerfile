# CPU-only image for reward-serve. The GPU on the host machine belongs to other
# work, so torch is installed from the CPU wheel index and CUDA_VISIBLE_DEVICES
# is emptied in the image itself, not just in the compose file.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HOME=/home/serve/.cache/huggingface

RUN groupadd --gid 10001 serve \
 && useradd --uid 10001 --gid 10001 --create-home serve

WORKDIR /app

COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY reward_serve /app/reward_serve
COPY scripts /app/scripts

USER serve

EXPOSE 8000

# /readyz is the right probe, not /healthz: /healthz answers as soon as the
# process is up, while /readyz only answers 200 once the model is loaded, the
# batching loop is running, and no drain has started.
HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=6 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status==200 else 1)"

CMD ["python", "-m", "reward_serve", "--host", "0.0.0.0", "--port", "8000"]
