FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ARG PYTHONPATH="tmp"

RUN : \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.11-venv \
        libglfw3 \
        libglfw3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.11 -m venv /venv --system-site-packages
ENV PATH=/venv/bin:$PATH

RUN python --version



COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 

RUN pip install fast-safe-rl --no-deps

COPY --link . /workspaces

RUN pip install -e /workspaces/envs/.

ENV PYTHONPATH=/workspaces:$PYTHONPATH

RUN mkdir /scratch

WORKDIR /workspaces