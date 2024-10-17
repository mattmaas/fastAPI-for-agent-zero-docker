from agent import Agent
from python.helpers import perplexity_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers import research_logger
import asyncio
import os

class OnlineKnowledge(Tool):
    async def execute(self, **kwargs):
        perplexity_result = await perplexity_search.perplexity_search(self.args["prompt"])
        duckduckgo_results = await asyncio.to_thread(duckduckgo_search.search, self.args["prompt"], results=5)
        
        # Log the full research results
        work_dir = self.agent.get_data("work_dir") or os.getcwd()
        research_logger.log_research(self.args["prompt"], perplexity_result["sources"], duckduckgo_results, work_dir)
        
        # Prepare the response for the agent
        response = f"Perplexity Answer: {perplexity_result['answer']}\n\nDuckDuckGo Results:\n"
        for result in duckduckgo_results[:5]:
            response += f"- {result['title']}: {result['href']}\n"
        
        return Response(
            message=response,
            break_loop=False,
        )
