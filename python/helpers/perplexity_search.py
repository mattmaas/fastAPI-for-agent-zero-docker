import os
import logging
import requests
from bs4 import BeautifulSoup
from python.tools import memory_tool
from openai import OpenAI
import models

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
    
    try:
        logging.info("Sending request to Perplexity API")
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
        logging.error(f"Error in Perplexity API call: {str(e)}")
        raise  # Re-raise the exception after logging
