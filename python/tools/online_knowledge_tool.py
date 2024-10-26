from agent import Agent
from python.helpers import perplexity_search
from python.tools import memory_tool
from python.helpers.tool import Tool, Response
from python.helpers import research_logger
import asyncio
import os

class OnlineKnowledge(Tool):
    async def execute(self, **kwargs):
        try:
            # Just do the perplexity search and memory recall
            perplexity_result = await perplexity_search.perplexity_search(self.args["prompt"])
            memories = await memory_tool.search(self.agent, self.args["prompt"])
            
            # Save the search result to memory
            await memory_tool.save(self.agent, f"Perplexity Search Result: {perplexity_result}")
            
            return Response(
                message=f"Perplexity Answer: {perplexity_result}\n\nRelated Memories:\n{memories}\n",
                break_loop=False
            )
        except Exception as e:
            logger.error(f"Error in OnlineKnowledge.execute: {str(e)}")
            return Response(message=f"Error occurred: {str(e)}", break_loop=False)
