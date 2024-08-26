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
    def execute(self, question="", **kwargs):
        logging.info(f"Knowledge tool executed with question: {question}")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            # perplexity search, if API provided
            if os.getenv("API_KEY_PERPLEXITY"):
                futures['perplexity'] = executor.submit(perplexity_search.perplexity_search, question)
            else: 
                PrintStyle.hint("No API key provided for Perplexity. Skipping Perplexity search.")

            # duckduckgo search
            futures['duckduckgo'] = executor.submit(duckduckgo_search.search, question)

            # memory search
            futures['memory'] = executor.submit(memory_tool.search, self.agent, question)

            # Wait for all functions to complete and handle any exceptions
            results = {}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                    logging.info(f"{name.capitalize()} search completed successfully")
                except Exception as e:
                    logging.error(f"Error in {name} search: {str(e)}")
                    results[name] = ""

        perplexity_result = results.get('perplexity', "")
        duckduckgo_result = results.get('duckduckgo', "")
        memory_result = results.get('memory', "")

        msg = files.read_file("prompts/tool.knowledge.response.md", 
                              online_sources = perplexity_result + "\n\n" + str(duckduckgo_result),
                              memory = memory_result )

        logging.info(f"Knowledge tool response prepared: {msg[:100]}...")  # Log first 100 characters of the response

        if self.agent.handle_intervention(msg): pass # wait for intervention and handle it, if paused

        return Response(message=msg, break_loop=False)
