# Base image with Python
FROM python:3.10-slim

# Metadata
LABEL maintainer="Snehil Kumar"
LABEL description="Dockerized version of py-simpace v2.0 - MRI motion artifact simulation with ML pipeline support."

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -e .

# Set default command
CMD ["/bin/bash"]
