import os
import logging
import json
import requests
from agent import Agent
from . import memory_tool
from python.helpers.tool import Tool, Response
from datetime import datetime
from python.helpers import files, perplexity_search
from python.helpers.print_style import PrintStyle
from python.helpers.research_logger import sanitize_filename

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class KnowledgeLite(Tool):
    async def execute(self):
        prompt = self.args.get("prompt")
        focus_mode = self.args.get("focus_mode") or "webSearch"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not prompt:
            return Response(message="No prompt provided for knowledge search", break_loop=False)

        try:
            work_dir = '/work_dir'

            # Perplexica search
            perplexica_result = self.perform_perplexica_search(prompt, focus_mode)

            # Perplexity search
            perplexity_result = await perplexity_search.perplexity_search(prompt)

            # Generate unique filename based on timestamp and query
            sanitized_query = sanitize_filename(prompt)[:100]
            filename = f"{self.timestamp}_{sanitized_query}.txt"
            research_file_path = os.path.join(work_dir, filename)

            # Prepare and save the research document
            research_document = (
                f"Research Query: {prompt}\n\n"
                f"Perplexica Summary:\n{perplexica_result['message']}\n\n"
                f"Perplexity Summary:\n{perplexity_result}\n\n"
                f"Source URLs:\n"
            )
            for source in perplexica_result['sources']:
                research_document += f"{source['metadata'].get('url', 'N/A')}\n"

            try:
                os.makedirs(os.path.dirname(research_file_path), exist_ok=True)
                with open(research_file_path, "w", encoding="utf-8") as f:
                    f.write(research_document)
                logger.info(f"Research document successfully saved to {research_file_path}")
                
                # Verify file was created
                if not os.path.exists(research_file_path):
                    raise FileNotFoundError(f"Failed to create research file at {research_file_path}")
                    
                file_size = os.path.getsize(research_file_path)
                logger.info(f"Research file size: {file_size} bytes")
            except Exception as e:
                logger.error(f"Error saving research document: {str(e)}")
                raise

            # Save results to memory
            await memory_tool.save(self.agent, f"Perplexica Search Answer for '{prompt}': {perplexica_result['message']}")
            await memory_tool.save(self.agent, f"Perplexity Search Answer for '{prompt}': {perplexity_result}")

            return Response(
                message=(
                    f"Perplexica Summary:\n{perplexica_result['message']}\n\n"
                    f"Perplexity Summary:\n{perplexity_result}\n\n"
                    f"Research has been saved to: {research_file_path}"
                ),
                break_loop=False
            )

        except Exception as e:
            error_message = f"Error in KnowledgeLite.execute: {str(e)}"
            logger.error(error_message)
            logger.exception("Full traceback:")
            return Response(message=f"An error occurred during knowledge search: {error_message}", break_loop=False)

    def perform_perplexica_search(self, query, focus_mode):
        try:
            base_url = os.getenv("PERPLEXICA_API_URL", "http://localhost:3001/api")
            url = f"{base_url}/search"
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
              "query": query
            })
            headers = {
              'Content-Type': 'application/json'
            }
            response = requests.post(url, headers=headers, data=payload, timeout=45)
            response.raise_for_status()

            result = response.json()
            return {
                "message": result.get("message", ""),
                "sources": result.get("sources", [])
            }

        except Exception as e:
            logger.error(f"Error in Perplexica search: {str(e)}")
            return {"message": "", "sources": []}
