import os
import logging
import requests
from bs4 import BeautifulSoup
from python.tools import memory_tool
from openai import OpenAI
import models

import os
import logging
import requests
from bs4 import BeautifulSoup
from python.tools import memory_tool
from openai import OpenAI
import models
import time

async def perplexity_search(query: str, agent=None, model_name="llama-3.1-sonar-large-128k-online", base_url="https://api.perplexity.ai") -> dict:
    logging.info(f"Perplexity search called with query: {query}")
    
    api_key = os.getenv("API_KEY_PERPLEXITY") or models.get_api_key("perplexity")
    if not api_key:
        raise ValueError("Perplexity API key not found in environment variables")

    logging.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")  # Log part of the API key for debugging

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    messages = [
        {
            "role": "system",
            "content": "Be thorough in your answer and provide as much relevant information as possible."
        },
        {
            "role": "user",
            "content": query
        },
    ]
    
    logging.info(f"Prepared messages: {messages}")
    
    retries = 3
    for attempt in range(retries):
        try:
            logging.info(f"Sending request to Perplexity API (Attempt {attempt + 1})")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            logging.info("Received response from Perplexity API")
            
            result = response.choices[0].message.content
            # Save the answer to memory
            if agent:
                await memory_tool.save(agent, f"Perplexity Search Answer for '{query}': {result}")

            logging.info(f"Extracted result: {result[:100]}...")  # Log first 100 characters of the result
            return {"answer": result, "sources": []}
        except Exception as e:
            logging.error(f"Error in Perplexity API call on attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries exceeded. Raising exception.")
                raise
```

python/helpers/knowledge_search.py
```python
<<<<< SEARCH
async def perform_perplexica_search(self, query, focus_mode):
    try:
        perplexica_url = os.getenv('PERPLEXICA_URL', 'http://100.108.162.61:3001')
        url = f"{perplexica_url}/api/search"
        data = {
            "chatModel": {
                "provider": "openai",
                "model": "gpt-4"
            },
            "embeddingModel": {
                "provider": "openai",
                "model": "text-embedding-3-large"
            },
            "optimizationMode": "balanced",
            "focusMode": focus_mode,
            "query": query
        }
        logger.debug(f"Attempting to connect to Perplexica at: {url}")
        logger.debug(f"With data: {data}")
        response = requests.post(url, json=data, timeout=180)
        response.raise_for_status()
        logger.debug("Successfully connected to Perplexica")

        result = response.json()
        
        # Save Perplexica sources to memory

        return result

    except Exception as e:
        logger.error(f"Error in Perplexica search: {str(e)}")
        return {"message": "", "sources": []}
