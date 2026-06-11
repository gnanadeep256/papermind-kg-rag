# Use a lightweight python image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install CPU-only PyTorch first to bypass heavy CUDA libraries
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch

# Install project dependencies plus UI/analysis libraries
RUN uv pip install --system --no-cache -r pyproject.toml \
    && uv pip install --system --no-cache streamlit pyvis pandas plotly networkx

# Copy the rest of the application
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Run the streamlit application
CMD ["streamlit", "run", "streamlit/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
