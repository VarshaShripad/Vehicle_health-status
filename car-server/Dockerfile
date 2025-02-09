# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application files
COPY . .

# Define environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8765

CMD ["python", "server.py"]