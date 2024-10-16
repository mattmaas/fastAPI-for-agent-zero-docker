from agent import Agent
from python.helpers import perplexity_search
from python.helpers.tool import Tool, Response
import asyncio

class OnlineKnowledge(Tool):
    async def execute(self, **kwargs):
        return Response(
            message=await process_prompt(self.args["prompt"]),
            break_loop=False,
        )

async def process_prompt(prompt):
    result = await asyncio.to_thread(perplexity_search.perplexity_search, prompt)
    return str(result)
