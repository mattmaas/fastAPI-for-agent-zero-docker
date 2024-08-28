# Use Python 3.10 or higher
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git

# Copy the entire repository into the container
COPY . /app

# Assuming your work_dir is in the root of your project
COPY ./work_dir /app/work_dir

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Load environment variables from .env file
ENV $(cat .env | xargs)

# Expose the port the app runs on
EXPOSE 8765

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]
