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

# Set default environment variables
ENV PERPLEXICA_API_URL="http://localhost:3001/api/search"
ENV API_KEY_OPENAI="your-default-openai-api-key"
ENV API_KEY_PERPLEXITY="your-default-perplexity-api-key"

# Expose the port the app runs on
EXPOSE 8766

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8766"]
