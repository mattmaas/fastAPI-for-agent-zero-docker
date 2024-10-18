import os
import logging
import requests
from python.tools import memory_tool
import models

async def perplexity_search(query: str, agent=None) -> dict:
    logging.info(f"Perplexity search called with query: {query}")
    
    api_key = os.getenv("API_KEY_PERPLEXITY") or models.get_api_key("perplexity")
    if not api_key:
        raise ValueError("Perplexity API key not found in environment variables")

    logging.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")  # Log part of the API key for debugging

    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": "Optional",
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        logging.info("Sending request to Perplexity API")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        logging.info("Received response from Perplexity API")
        
        result = response.json()
        answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Save the answer to memory
        if agent:
            await memory_tool.save(agent, f"Perplexity Search Answer for '{query}': {answer}")

        logging.info(f"Extracted result: {answer[:100]}...")  # Log first 100 characters of the result
        return {"answer": answer, "sources": result.get('sources', [])}
    except Exception as e:
        logging.error(f"Error in Perplexity API call: {str(e)}")
        raise

python/helpers/knowledge_search.py
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
