from agent import Agent
from python.helpers import perplexity_search
from python.helpers.tool import Tool, Response
from python.helpers import research_logger
import asyncio
import os

class OnlineKnowledge(Tool):
    async def execute(self, **kwargs):
        perplexity_result = await perplexity_search.perplexity_search(self.args["prompt"])
        
        # Log the full research results
        work_dir = self.agent.get_data("work_dir") or os.getcwd()
        research_logger.log_research(self.args["prompt"], perplexity_result["sources"], work_dir)
        
        # Prepare the response for the agent
        response = f"Perplexity Answer: {perplexity_result['answer']}\n"
        
        return Response(
            message=response,
            break_loop=False,
        )
