import os
import logging
import json
import requests
import asyncio
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
from PIL import Image
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
    async def execute(self):
        prompt = self.args.get("prompt")
        focus_mode = self.args.get("focus_mode") or "webSearch"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.debug(f"Knowledge.execute called with prompt: {prompt}, focus_mode: {focus_mode}")
        if not prompt:
            logger.warning("No prompt provided for knowledge search")
            return Response(message="No prompt provided for knowledge search", break_loop=False)

        try:
            work_dir = self.agent.get_data("work_dir") or os.getcwd()

            # Step 1: Perplexity search
            logger.debug("Starting Perplexity search")
            perplexity_result = await perplexity_search.perplexity_search(prompt)
            logger.debug(f"Perplexity search completed")
            await memory_tool.save(self.agent, f"Perplexity Summary for '{prompt}': {perplexity_result}")

            # Step 2: Perplexica search
            logger.debug("Starting Perplexica search")
            perplexica_result = self.perform_perplexica_search(prompt, focus_mode)
            perplexica_summary = perplexica_result['message']
            perplexica_sources = perplexica_result['sources']
            logger.debug(f"Perplexica search completed")
            await memory_tool.save(self.agent, f"Perplexica Initial Summary for '{prompt}': {perplexica_summary}")

            # Step 3: Fetch and summarize full source contents
            logger.debug("Processing source contents")
            source_contents = []
            source_urls = []
            for source in perplexica_sources:
                url = source['metadata'].get('url', '')
                title = source['metadata'].get('title', 'N/A')
                full_text = self.fetch_full_content(url)
                source_contents.append({
                    'url': url,
                    'title': title,
                    'content': full_text
                })
                source_urls.append(url)

            # Generate source content summary using GPT-4
            sources_summary_prompt = (
                "Please provide a comprehensive summary of these source materials. Focus on:\n"
                "- Key findings and main points from each source\n"
                "- Important data points and statistics\n"
                "- Methodologies and approaches discussed\n"
                "- Notable examples and case studies\n"
                "- Relationships between different sources\n\n"
                "Sources:\n"
            )
            for source in source_contents:
                sources_summary_prompt += f"\nSource: {source['title']}\n{source['content']}\n"

            sources_summary = await self.agent.send_adhoc_message(
                system="You are a research assistant tasked with creating detailed, accurate summaries of source materials.",
                msg=sources_summary_prompt,
                output_label="Generating source content summary"
            )
            await memory_tool.save(self.agent, f"Source Content Summary for '{prompt}': {sources_summary}")

            # Generate executive summary using o1-preview model
            executive_summary_prompt = (
                "Please provide an executive summary synthesizing these three summaries:\n\n"
                f"Perplexity Summary:\n{perplexity_result}\n\n"
                f"Perplexica Summary:\n{perplexica_summary}\n\n"
                f"Source Content Summary:\n{sources_summary}\n\n"
                "Focus on:\n"
                "- Key findings and conclusions\n"
                "- Specific data points and statistics\n"
                "- Methodologies and approaches\n"
                "- Important relationships and correlations\n"
                "- Concrete examples and case studies"
            )

            # Create temporary config for o1-preview model
            o1_config = self.agent.config.copy()
            o1_config.chat_model = get_openai_chat(model_name="o1-preview", api_key=None)
            
            executive_summary = await self.agent.send_adhoc_message(
                system="You are an executive assistant tasked with creating concise, comprehensive summaries.",
                msg=executive_summary_prompt,
                output_label="Generating executive summary",
                config=o1_config
            )
            await memory_tool.save(self.agent, f"Executive Summary for '{prompt}': {executive_summary}")

            # Generate unique filename and save research document
            sanitized_query = sanitize_filename(prompt)[:100]
            filename = f"{self.timestamp}_{sanitized_query}.txt"
            research_file_path = os.path.join(work_dir, filename)

            # Prepare research document in specified order
            research_document = (
                f"Research Document\n\n"
                f"Executive Summary:\n{executive_summary}\n\n"
                f"Perplexica Initial Summary:\n{perplexica_summary}\n\n"
                f"Perplexity Summary:\n{perplexity_result}\n\n"
                f"Source Content Summary:\n{sources_summary}\n\n"
                f"Source URLs:\n" + "\n".join(source_urls) + "\n\n"
                f"Full Source Contents:\n"
            )
            
            for source in source_contents:
                research_document += f"\n=== Source: {source['title']} ===\n"
                research_document += f"URL: {source['url']}\n\n"
                research_document += f"Content:\n{source['content']}\n\n"

            with open(research_file_path, "w", encoding="utf-8") as f:
                f.write(research_document)

            logger.debug(f"Research document saved to {research_file_path}")

            # Return executive summary with file info
            final_response = f"{executive_summary}\n\nResearch document saved to: {research_file_path}"
            return Response(message=final_response, break_loop=False)

        except Exception as e:
            error_message = f"Error in Knowledge.execute: {str(e)}"
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
              "query": query,
              "history": [
                [
                  "human",
                  "Hi, how are you?"
                ],
                [
                  "assistant",
                  "I am doing well, how can I help you today?"
                ]
              ]
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

    def prepare_research_document(self, perplexity_answer, perplexica_sources, perplexica_message, memories, research_file_path):
        # First prepare the executive summary using the utility model
        summarization_prompt = (
            "Please provide a comprehensive executive summary of the following research content. "
            "Focus on:\n"
            "- Key findings and conclusions\n"
            "- Specific data points and statistics\n"
            "- Methodologies and approaches used\n"
            "- Important relationships and correlations\n"
            "- Concrete examples and case studies\n\n"
            f"Perplexity Answer:\n{perplexity_answer}\n\n"
            f"Perplexica Summary:\n{perplexica_message}\n\n"
            f"Related Memories:\n{memories}\n"
        )
        
        executive_summary = asyncio.run(self.agent.send_adhoc_message(
            system="You are a research assistant tasked with creating detailed, accurate executive summaries.",
            msg=summarization_prompt,
            output_label="Generating executive summary"
        ))

        document = (
            f"Research Document\n\n"
            f"Executive Summary:\n{executive_summary}\n\n"
            f"Perplexity Answer:\n{perplexity_answer}\n\n"
            f"Perplexica Summary:\n{perplexica_message}\n\n"
            f"Related Memories:\n{memories}\n\n"
            f"Source Contents:\n"
        )
        
        for source in perplexica_sources:
            url = source['metadata'].get('url', '')
            title = source['metadata'].get('title', 'N/A')
            full_text = self.fetch_full_content(url, research_file_path)
            document += f"\n=== Source: {title} ===\nURL: {url}\n\nDetailed Content:\n{full_text}\n\n"

        return document

    async def prepare_agent_message(self, perplexity_answer, perplexica_summary, research_file_path):
        # First, read the research document
        with open(research_file_path, 'r', encoding='utf-8') as f:
            research_content = f.read()
        
        # Prepare detailed summarization instructions
        summarization_prompt = (
            "Please provide a comprehensive synthesis of the following research content. "
            "Focus on:\n"
            "- Preserving specific details, numerical data, and key findings\n"
            "- Extracting and combining core insights from all sources\n"
            "- Maintaining precision and depth of the original information\n"
            "- Key findings and conclusions\n"
            "- Specific data points and statistics\n"
            "- Methodologies and approaches used\n"
            "- Important relationships and correlations\n"
            "- Concrete examples and case studies\n\n"
            f"Research Content:\n{research_content}"
        )
        
        # Get the summary using the utility model
        summary = await self.agent.send_adhoc_message(
            system="You are a research assistant tasked with creating detailed, accurate summaries.",
            msg=summarization_prompt,
            output_label="Generating research summary"
        )
        
        # Return the formatted response
        return files.read_file("prompts/tool.knowledge.response.md",
                             combined_result=summary,
                             research_file_path=research_file_path)
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
