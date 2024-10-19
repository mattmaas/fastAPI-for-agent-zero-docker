import os
import logging
import json
import requests
import asyncio
from bs4 import BeautifulSoup
from agent import Agent
from . import memory_tool
from python.helpers.tool import Tool, Response
from datetime import datetime
from python.helpers import files, perplexity_search
from python.helpers.print_style import PrintStyle
from python.helpers.research_logger import sanitize_filename

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Knowledge(Tool):
    async def execute(self, **kwargs):
        prompt = kwargs.get("prompt") or self.args.get("prompt")
        focus_mode = kwargs.get("focus_mode") or self.args.get("focus_mode") or "webSearch"
        logger.debug(f"Knowledge.execute called with prompt: {prompt}, focus_mode: {focus_mode}")
        if not prompt:
            logger.warning("No prompt provided for knowledge search")
            return Response(message="No prompt provided for knowledge search", break_loop=False)

        try:
            work_dir = self.agent.get_data("work_dir") or os.getcwd()

            # Perplexica search
            logger.debug("Starting Perplexica search")
            perplexica_result = await self.perform_perplexica_search(prompt, focus_mode)
            logger.debug(f"Perplexica search completed. Result length: {len(perplexica_result['message'])}")

            # Perplexity search
            logger.debug("Starting Perplexity search")
            perplexity_result = await perplexity_search.perplexity_search(prompt)
            logger.debug(f"Perplexity search completed. Answer length: {len(perplexity_result['answer'])}")

            # Prepare the research document
            research_document = self.prepare_research_document(perplexity_result['answer'], perplexica_result['sources'])

            # Generate unique filename based on timestamp and query
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_query = sanitize_filename(prompt)[:100]
            filename = f"{timestamp}_{sanitized_query}.txt"
            research_file_path = os.path.join(work_dir, filename)

            # Save the research document
            with open(research_file_path, "w", encoding="utf-8") as f:
                f.write(research_document)

            logger.debug(f"Research document saved to {research_file_path}")

            # Save Perplexica result to memory
            await memory_tool.save(self.agent, f"Perplexica Search Answer for '{prompt}': {perplexica_result['message']}")

            # Save full contents of Perplexica sources to memory

            # Save Perplexity result to memory
            await memory_tool.save(self.agent, f"Perplexity Search Answer for '{prompt}': {perplexity_result['answer']}")

            # Prepare the message for the agent
            logger.debug("Preparing agent message")
            perplexica_summary = perplexica_result['message']
            perplexity_answer = perplexity_result['answer']
            msg = await self.prepare_agent_message(perplexity_answer, perplexica_summary, research_file_path)
            logger.debug(f"Agent message prepared. Length: {len(msg)}")

            if await self.agent.handle_intervention(msg): 
                logger.debug("Agent intervention handled")
            else:
                logger.debug("No agent intervention required")

            logger.debug("Knowledge.execute completed successfully")
            return Response(message=msg, break_loop=False)

        except Exception as e:
            error_message = f"Error in Knowledge.execute: {str(e)}"
            logger.error(error_message)
            logger.exception("Full traceback:")
            return Response(message=f"An error occurred during knowledge search: {error_message}", break_loop=False)

    async def perform_perplexica_search(self, query, focus_mode):
        try:
            url = "http://100.108.162.61:3001/api/search"
            payload = json.dumps({
              "chatModel": {
                "provider": "openai",
                "model": "gpt-4o"
              },
              "embeddingModel": {
                "provider": "openai",
                "model": "text-embedding-3-large"
              },
              "optimizationMode": "speed",
              "focusMode": focus_mode,
              "query": query,
            })
            headers = {
              'Content-Type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            logger.debug("Successfully connected to Perplexica")

            result = response.json()
            
            # Save Perplexica sources to memory

            return result

        except Exception as e:
            logger.error(f"Error in Perplexica search: {str(e)}")
            return {"message": "", "sources": []}

    def prepare_research_document(self, perplexity_answer, sources):
        document = f"Perplexity Answer:\n{perplexity_answer}\n\n"
        document += "Research Document\n\n"
        document += "Perplexica Sources:\n"
        for source in sources:
            url = source['metadata'].get('url', '')
            full_text = self.fetch_full_content(url)
            document += f"Title: {source['metadata'].get('title', 'N/A')}\n"
            document += f"URL: {url}\n"
            document += f"Full Content:\n{full_text}\n\n"

        return document

    async def prepare_agent_message(self, perplexity_answer, perplexica_summary, research_file_path):
        return files.read_file("prompts/tool.knowledge.response.md", 
                               perplexity_answer=perplexity_answer,
                               perplexica_summary=perplexica_summary,
                               research_file_path=research_file_path)
    def fetch_full_content(self, url):
        try:
            if not url:
                return "No URL provided."
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error fetching content from {url}: {str(e)}"
