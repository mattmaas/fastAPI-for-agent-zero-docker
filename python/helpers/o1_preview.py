from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

async def generate_executive_summary(original_query: str, summaries_prompt: str) -> str:
    """Generate an executive report using OpenAI's o1-preview model"""
    try:
        client = OpenAI(api_key=os.getenv("API_KEY_OPENAI"))
        
        # Combine instruction with prompts since o1-preview doesn't support system messages
        enhanced_prompt = (
            "Create a comprehensive executive summary that:\n"
            "1. Addresses the original query directly and thoroughly\n"
            "2. Provides a detailed synthesis of all key points and major findings from each research source\n"
            "3. Integrates relevant insights from your knowledge\n\n"
            f"Original Query:\n{original_query}\n\n"
            "Research Summaries to Synthesize:\n"
            f"{summaries_prompt}"
        )
        
        response = client.chat.completions.create(
            model="o1-preview",
            messages=[
                {"role": "user", "content": enhanced_prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating executive summary with o1-preview: {str(e)}")
        raise
