FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including espeak-ng
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    curl \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for sudachipy)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with error checking
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install uvicorn

COPY . .

EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]