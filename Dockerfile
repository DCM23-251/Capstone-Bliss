# Start with a robust Python base image
FROM python:3.10

# Install system dependencies needed for the headless environment and general Python work
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libglib2.0-0 \
    libgl1 \
    # Clean up apt lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# --- CRITICAL STEP: Force headless installation first ---
# This command installs opencv-python-headless and tells pip to ignore dependency checks
# for this specific step, preventing 'ultralytics' from forcing a conflict initially.
RUN pip install --no-cache-dir \
    "opencv-python-headless" \
    --force-reinstall \
    --no-deps

# Install all remaining dependencies from requirements.txt
# Pip will now see opencv-python-headless is already installed and satisfied.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Define the command to run your application (modify "app.py" as needed)
CMD ["python", "main.py"]
