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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Knowledge(Tool):
    async def execute(self, prompt="", **kwargs):
        logger.debug(f"Knowledge.execute called with prompt: {prompt}")
        if not prompt:
            prompt = self.args.get("prompt", "")
        if not prompt:
            logger.warning("No prompt provided for knowledge search")
            return Response(message="No prompt provided for knowledge search", break_loop=False)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Schedule the functions to be run in parallel
                futures = {}

                # perplexity search, if API provided
                if os.getenv("API_KEY_PERPLEXITY"):
                    futures['perplexity'] = executor.submit(perplexity_search.perplexity_search, prompt)
                else: 
                    logger.info("No API key provided for Perplexity. Skipping Perplexity search.")
                
                # duckduckgo search
                futures['duckduckgo'] = executor.submit(duckduckgo_search.search, prompt)

                # memory search
                futures['memory'] = executor.submit(memory_tool.search, self.agent, prompt)

                # Wait for all functions to complete
                concurrent.futures.wait(futures.values())

                results = {}
                for key, future in futures.items():
                    try:
                        results[key] = future.result()
                    except Exception as e:
                        logger.error(f"Error in {key} search: {str(e)}")
                        results[key] = f"Error occurred during {key} search"

            perplexity_result = results.get('perplexity', "")
            duckduckgo_result = results.get('duckduckgo', "")
            memory_result = results.get('memory', "")

            msg = files.read_file("prompts/tool.knowledge.response.md", 
                                  online_sources = f"{perplexity_result}\n\n{duckduckgo_result}",
                                  memory = memory_result)

            if self.agent.handle_intervention(msg): 
                pass  # wait for intervention and handle it, if paused

            logger.debug(f"Knowledge.execute completed successfully")
            return Response(message=msg, break_loop=False)

        except Exception as e:
            logger.error(f"Error in Knowledge.execute: {str(e)}")
            return Response(message=f"An error occurred during knowledge search: {str(e)}", break_loop=False)
