# Use Python 3.9 image from the official Docker repository
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the local code into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port that Flask will run on
EXPOSE 5000

# Run the app using Gunicorn to handle requests more efficiently in production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

