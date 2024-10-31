import os
from datetime import datetime
import re

def sanitize_filename(filename):
    # Remove invalid characters
    sanitized = re.sub(r'[^\w\-_\. ]', '_', filename)
    return sanitized

def log_research(query, answer, work_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_query = sanitize_filename(query)[:100]  # Truncate to 100 characters
    filename = f"{timestamp}_{sanitized_query}.txt"

    if work_dir is None:
        work_dir = os.getcwd()  # Use current working directory if work_dir is None
    research_dir = os.path.join(work_dir, "research")
    os.makedirs(research_dir, exist_ok=True)

    filepath = os.path.join(research_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Main Query: {query}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Perplexity Answer:\n")
        f.write(f"{answer}\n")

    return filepath

