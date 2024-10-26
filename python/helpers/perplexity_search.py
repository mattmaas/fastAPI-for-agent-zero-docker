from openai import OpenAI
import models
import logging
import re

def strip_ansi_codes(text):
    """Remove ANSI escape sequences from text while preserving content"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

async def perplexity_search(query: str, model_name="llama-3.1-sonar-large-128k-online", api_key=None, base_url="https://api.perplexity.ai"):
    logging.info(f"Perplexity search called with query: {strip_ansi_codes(query)}")
    
    api_key = api_key or models.get_api_key("perplexity")
    logging.info("Using Perplexity API key")  # Don't log any part of the API key

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
        clean_result = strip_ansi_codes(result)
        logging.info(f"Extracted result: {clean_result[:100]}...")  # Log first 100 characters of the result
        return result  # Return original result to preserve formatting for display
    except Exception as e:
        logging.error(f"Error in Perplexity API call: {str(e)}")
        raise  # Re-raise the exception after logging
