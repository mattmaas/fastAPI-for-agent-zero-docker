# Use Python 3.10 or higher
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOME_ASSISTANT_URL=http://host.docker.internal:8123
ENV HOME_ASSISTANT_ACCESS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4NWQzZWIwYWIxMDc0NDM3YWQxNmNmNmE3OGE4OTgyYSIsImlhdCI6MTcyMzg3NTMyMywiZXhwIjoyMDM5MjM1MzIzfQ.V6MrwmtS5rb2VJ7S3N2UoN2c8GI1bpcCIZ_oYTJZTFI

# Set work directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y git

# Copy the entire repository into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8765

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8765"]
