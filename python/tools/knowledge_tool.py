import os
import logging
from agent import Agent
from . import online_knowledge_tool
from python.helpers import perplexity_search
from python.helpers import duckduckgo_search

from . import memory_tool
import concurrent.futures

from python.helpers.tool import Tool, Response
from python.helpers import files
from python.helpers.print_style import PrintStyle

class Knowledge(Tool):
    def execute(self, prompt="", **kwargs):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Schedule all search functions to run in parallel
                perplexity_future = executor.submit(perplexity_search.perplexity_search, prompt)
                duckduckgo_future = executor.submit(duckduckgo_search.search, prompt)
                memory_future = executor.submit(memory_tool.search, self.agent, prompt)

                # Wait for all functions to complete
                perplexity_result = perplexity_future.result()
                duckduckgo_result = duckduckgo_future.result()
                memory_result = memory_future.result()

            online_sources = f"Perplexity: {perplexity_result}\n\nDuckDuckGo: {duckduckgo_result}"

            msg = files.read_file("prompts/tool.knowledge.response.md", 
                                  online_sources=online_sources,
                                  memory=memory_result)

            if self.agent.handle_intervention(msg): pass # wait for intervention and handle it, if paused

            return Response(message=msg, break_loop=False)
        except Exception as e:
            logging.error(f"Error in Knowledge.execute: {str(e)}", exc_info=True)
            return Response(message=f"An error occurred during knowledge gathering: {str(e)}", break_loop=True)
