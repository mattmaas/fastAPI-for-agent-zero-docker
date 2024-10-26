import os
import logging
import json
import requests
import asyncio
from openai import OpenAI
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
from PIL import Image
from agent import Agent
from . import memory_tool
from python.helpers.tool import Tool, Response
from datetime import datetime
from python.helpers import files, perplexity_search, o1_preview
from python.helpers.print_style import PrintStyle
from python.helpers.research_logger import sanitize_filename

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Knowledge(Tool):
    async def execute(self):
        prompt = self.args.get("prompt")
        focus_mode = self.args.get("focus_mode") or "webSearch"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.debug(f"Knowledge.execute called with prompt: {prompt}, focus_mode: {focus_mode}")
        if not prompt:
            logger.warning("No prompt provided for knowledge search")
            return Response(message="No prompt provided for knowledge search", break_loop=False)

        try:
            work_dir = '/work_dir'

            # Perplexica search
            logger.debug("Starting Perplexica search")
            perplexica_result = self.perform_perplexica_search(prompt, focus_mode)
            logger.debug(f"Perplexica search completed. Result length: {len(perplexica_result['message'])}")

            # Perplexity search
            logger.debug("Starting Perplexity search")
            perplexity_result = await perplexity_search.perplexity_search(prompt)
            logger.debug(f"Perplexity search completed. Answer length: {len(perplexity_result)}")

            # Fetch related memories
            memories = await memory_tool.search(self.agent, prompt)
            logger.debug(f"Fetched memories. Count: {len(memories)}")

            # Generate unique filename based on timestamp and query
            sanitized_query = sanitize_filename(prompt)[:100]
            filename = f"{self.timestamp}_{sanitized_query}.txt"
            research_file_path = os.path.join(work_dir, filename)

            # Generate source content summary using GPT-4
            sources_summary_prompt = files.read_file("prompts/tool.knowledge.source_summary.md")
            sources_content = ""
            for source in perplexica_result['sources']:
                url = source['metadata'].get('url', '')
                full_text = self.fetch_full_content(url, research_file_path)
                sources_content += f"\nSource: {source['metadata'].get('title', 'N/A')}\n{full_text}\n"

            # Use o1-mini model specifically for source content summarization
            client = OpenAI(api_key=os.getenv("API_KEY_OPENAI"))
            response = client.chat.completions.create(
                model="o1-mini",
                messages=[
                    {"role": "system", "content": "You are a research assistant tasked with creating detailed, accurate summaries of source materials."},
                    {"role": "user", "content": sources_summary_prompt + "\n\nSources:\n" + sources_content}
                ]
            )
            sources_summary = response.choices[0].message.content

            # Generate executive report
            summaries_prompt = (
                f"Perplexity Summary:\n{perplexity_result}\n\n"
                f"Perplexica Summary:\n{perplexica_result['message']}\n\n"
                f"Source Content Summary:\n{sources_summary}\n\n"
            )

            executive_summary = await o1_preview.generate_executive_summary(prompt, summaries_prompt)

            # Prepare and save the research document
            research_document = self.prepare_research_document(
                perplexity_result,
                perplexica_result['sources'],
                perplexica_result['message'],
                memories,
                research_file_path,
                sources_summary,
                executive_summary,
                prompt
            )

            # Save the research document
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

            # Save Perplexica initial result to memory
            await memory_tool.save(self.agent, f"Perplexica Search Answer for '{prompt}': {perplexica_result}")
            
            # Save Perplexity result to memory
            await memory_tool.save(self.agent, f"Perplexity Search Answer for '{prompt}': {perplexity_result}")


            # Prepare the message for the agent
            logger.debug("Preparing agent message")
            perplexica_summary = perplexica_result['message']
            perplexity_answer = perplexity_result
            msg = await self.prepare_agent_message(perplexity_answer, perplexica_summary, research_file_path, executive_summary)
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
            return Response(message=f"An error occurred during knowledge search: {str(e)}", break_loop=False)

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
            logger.debug("Successfully connected to Perplexica")

            result = response.json()
            message = result.get("message", "")
            sources = result.get("sources", [])

            logger.debug(f"Perplexica message: {message[:100]}...")  # Log first 100 characters of the message
            logger.debug(f"Number of sources: {len(sources)}")

            return {"message": message, "sources": sources}

        except Exception as e:
            logger.error(f"Error in Perplexica search: {str(e)}")
            return {"message": "", "sources": []}

    def prepare_research_document(self, perplexity_answer, perplexica_sources, perplexica_message, memories, research_file_path, sources_summary, executive_summary, prompt):
        document = f"Research Query: {prompt}\n\n"
        document += f"Executive Summary:\n{executive_summary}\n\n"
        document += f"Perplexica Summary:\n{perplexica_message}\n\n"
        document += f"Perplexity Summary:\n{perplexity_answer}\n\n"
        document += f"Source Content Summary:\n{sources_summary}\n\n"
        document += f"Related Memories:\n{memories}\n\n"
        document += f"Source URLs:\n"
        for source in perplexica_sources:
            document += f"{source['metadata'].get('url', 'N/A')}\n"
        document += "\nFull Source Contents:\n"
        for source in perplexica_sources:
            url = source['metadata'].get('url', '')
            full_text = self.fetch_full_content(url, research_file_path)
            document += f"Title: {source['metadata'].get('title', 'N/A')}\n"
            document += f"URL: {url}\n"
            document += f"Full Content:\n{full_text}\n\n"

        return document

    async def prepare_agent_message(self, perplexity_answer, perplexica_summary, research_file_path, executive_summary):
        return f"Executive Summary:\n{executive_summary}\n\nDetailed research has been saved to: {research_file_path}\n\n"
        
    def fetch_full_content(self, url, research_file_path):
        try:
            if not url:
                return "No URL provided."
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Check if the content is a PDF
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_file = BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""

                for page in pdf_reader.pages:
                    text += page.extract_text() or ""

                return text

            # If not a PDF, treat as HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error fetching content from {url}: {str(e)}"
