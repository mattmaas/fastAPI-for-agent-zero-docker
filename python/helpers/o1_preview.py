from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ExecutiveSummaryError(Exception):
    """Custom exception for executive summary generation errors"""
    pass

async def generate_executive_summary(prompt: str, max_tokens: Optional[int] = 1000) -> str:
    """Generate an executive summary using OpenAI's o1-preview model
    
    Args:
        prompt: The text to summarize
        max_tokens: Maximum length of the generated summary
        
    Returns:
        str: The generated executive summary
        
    Raises:
        ExecutiveSummaryError: If there's an error generating the summary
    """
    api_key = os.getenv("API_KEY_OPENAI")
    if not api_key:
        raise ExecutiveSummaryError("OpenAI API key not found in environment variables")
        
    try:
        client = OpenAI(api_key=api_key)
        
        # Combine instruction with prompt since o1-preview doesn't support system messages
        enhanced_prompt = "As an executive assistant, create a concise, comprehensive summary.\n\n" + prompt
        
        response: ChatCompletion = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        if not response.choices or not response.choices[0].message:
            raise ExecutiveSummaryError("No response received from OpenAI API")
            
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Error generating executive summary with o1-preview: {str(e)}"
        logger.error(error_msg)
        raise ExecutiveSummaryError(error_msg) from e
