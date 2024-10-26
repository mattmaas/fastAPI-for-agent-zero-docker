from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

async def generate_executive_summary(prompt: str) -> str:
    """Generate an executive summary using OpenAI's o1-preview model"""
    try:
        client = OpenAI(api_key=os.getenv("API_KEY_OPENAI"))
        
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {"role": "system", "content": "You are an executive assistant tasked with creating concise, comprehensive summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating executive summary with o1-preview: {str(e)}")
        raise
