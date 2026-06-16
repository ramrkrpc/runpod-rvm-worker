FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod "av==9.2.0" requests pims tqdm

# pre-clone RVM so cold start is faster
RUN git clone --depth 1 https://github.com/PeterL1n/RobustVideoMatting /RVM

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
