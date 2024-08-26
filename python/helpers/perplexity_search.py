
from openai import OpenAI
import models
import logging

def perplexity_search(query:str, model_name="llama-3.1-sonar-large-128k-online",api_key=None,base_url="https://api.perplexity.ai"):    
    logging.info(f"Perplexity search called with query: {query}")
    
    api_key = api_key or models.get_api_key("perplexity")
    logging.info(f"Using API key: {api_key[:5]}...{api_key[-5:]}")  # Log part of the API key for debugging

    client = OpenAI(api_key=api_key, base_url=base_url)
        
    messages = [
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
        logging.info(f"Extracted result: {result[:100]}...")  # Log first 100 characters of the result
        return result
    except Exception as e:
        logging.error(f"Error in Perplexity API call: {str(e)}")
        raise  # Re-raise the exception after logging
