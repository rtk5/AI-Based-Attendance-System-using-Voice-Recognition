# Use official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Unzip the recordings if needed (optional)
RUN apt-get update && apt-get install -y unzip && \
    unzip all_voice_recordings.zip -d recordings || true

# Default command to run
CMD ["python", "main1.py"]
