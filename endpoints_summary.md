# API Endpoints Summary

## /research

The `/research` endpoint is handled by the `Knowledge` tool defined in `python/tools/knowledge_tool.py`. It performs comprehensive research based on a given prompt, including:

- Performing Perplexica search to get detailed summaries and sources.
- Performing Perplexity search using the Perplexity API.
- Fetching related memories from the agent's memory.
- Generating an executive summary using OpenAI models.
- Preparing and saving a detailed research document.
- Saving results to memory for future recall.

## /research-lite

The `/research-lite` endpoint is handled by the `KnowledgeLite` tool defined in `python/tools/knowledge_lite_tool.py`. It provides a lightweight research process that includes:

- Performing Perplexica and Perplexity searches.
- Generating an executive summary using OpenAI models.
- Saving a summarized research document.
- Saving search results to memory.

## /perplexity

The `/perplexity` endpoint utilizes the `perplexity_search` function defined in `python/helpers/perplexity_search.py`. It performs searches using the Perplexity API and returns responses based on the provided prompt.

- Communicates with the Perplexity API using OpenAI client.
- Handles messages and extracts responses.
- Logs and handles any errors during the API calls.

## /research-import

The `/research-import` endpoint is handled by the `ResearchImport` tool defined in `python/tools/research_import_tool.py`. It allows importing existing research documents into the agent's memory.

- Scans a specified directory for `.txt` research files.
- Reads and processes each research document.
- Generates concise memory prompts using OpenAI models.
- Saves the content and prompts to the agent's memory.
