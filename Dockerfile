FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set MPLCONFIGDIR to writable location
ENV MPLCONFIGDIR=/tmp/matplotlib

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Run Flask
CMD ["python", "app.py"]
