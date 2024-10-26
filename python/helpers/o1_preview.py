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
            "As a research analyst, create an executive report that:\n"
            "1. Analyzes and synthesizes the following research summaries\n"
            "2. Specifically addresses the original research query\n"
            "3. Provides insights and conclusions based on both the research data and your knowledge\n\n"
            f"Original Research Query:\n{original_query}\n\n"
            "Research Summaries to Analyze:\n"
            f"{summaries_prompt}\n\n"
            "Format your response as a professional executive report with clear sections for:\n"
            "- Key Findings\n"
            "- Analysis\n"
            "- Recommendations/Conclusions"
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
