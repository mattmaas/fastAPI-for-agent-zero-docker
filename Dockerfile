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
RUN pip install -r requirements.txt

# Set default environment variables
ENV PERPLEXICA_API_URL="http://localhost:3001/api/search"
ENV API_KEY_OPENAI="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"
ENV API_KEY_PERPLEXITY="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"

# Expose the port the app runs on
EXPOSE 8766

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8766"]
