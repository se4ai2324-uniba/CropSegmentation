# syntax=docker/dockerfile:1
ARG REMOTE_DATA
ARG REMOTE_MODEL
# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.11.4
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /

# Fix: ERROR failed building wheel for psutil
RUN apt-get update -y && apt-get install gcc ffmpeg libsm6 libxext6 wget unzip -y


# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements-prod.txt,target=requirements-prod.txt \
    python -m pip install -r requirements-prod.txt


# Copy the source code into the container.
COPY . .

ARG REMOTE_DATA
ARG REMOTE_MODEL
ENV REMOTE_DATA=${REMOTE_DATA}
ENV REMOTE_MODEL=${REMOTE_MODEL}
RUN wget -P data/processed ${REMOTE_DATA}
RUN wget -P models/saved ${REMOTE_MODEL}
RUN unzip -o data/processed/datasets_processed.zip

WORKDIR /src/api

# Expose the port that the application listens on.
EXPOSE 80

# Run the application.
CMD uvicorn server:app --host 0.0.0.0 --port 80
