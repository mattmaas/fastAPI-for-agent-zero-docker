# API Endpoints Summary

## /research

The `/research` endpoint is handled by the `Knowledge` tool defined in `python/tools/knowledge_tool.py`. It performs comprehensive research based on a given prompt, engaging in an extensive process that includes:

- Performing **Perplexica** search to obtain detailed summaries and source URLs.
  - **Perplexica** is an advanced search service that retrieves and summarizes relevant information from the web.
- Fetching full source contents from the web and using the **o1-mini** model to extract important details with an intelligent prompt.
  - This involves parsing websites, PDFs, and other media to gather comprehensive data.
- Performing **Perplexity** search using the Perplexity API.
  - **Perplexity** is an AI-powered search engine that provides concise answers and summaries from the web.
- Combining all gathered information into a detailed executive summary using OpenAI's advanced models.
  - The executive summary synthesizes Perplexica and Perplexity answers along with the extracted source details into a cohesive report.
- Preparing and saving a comprehensive research document that includes the executive summary, detailed findings, and source references.
- Saving key findings and summaries to the memory bank for future use and prompting.
  - This enhances the agent's ability to recall and utilize past research in future interactions.

## /research-lite

The `/research-lite` endpoint is handled by the `KnowledgeLite` tool defined in `python/tools/knowledge_lite_tool.py`. It offers a streamlined research process distinct from `/research`, focusing on:

- Performing **Perplexica** and **Perplexity** searches to quickly gather relevant information.
- Generating an executive summary using OpenAI models without fetching full source contents.
  - It uses the summaries from Perplexica and Perplexity directly.
- Preparing and saving a concise research document that includes the executive summary and key findings.
- Saving the executive summary and key findings to the memory bank for future reference.
  - Ideal for situations requiring faster results with less computational overhead.

## /perplexity

The `/perplexity` endpoint utilizes the `perplexity_search` function defined in `python/helpers/perplexity_search.py`. It performs searches using the **Perplexity** API and returns concise answers based on the provided prompt.

- **Perplexity** is an AI-powered search engine that provides summarized information and direct answers from the web.
- Communicates with the Perplexity API to fetch real-time information.
- Handles messages, extracts responses, and logs interactions.
- Saves the search results to the memory bank for future use and prompting.
  - This allows the agent to reference Perplexity's insights in subsequent tasks.
- Useful for obtaining quick, succinct answers to specific queries without extensive processing.

## /research-import

The `/research-import` endpoint is handled by the `ResearchImport` tool defined in `python/tools/research_import_tool.py`. It allows importing existing research documents into the agent's memory bank.

- Scans a specified directory for `.txt` research files.
- Reads and processes each research document.
- Uses OpenAI models to generate concise memory prompts that capture the essence of each document.
- Saves both the original content and the generated prompts to the agent's memory bank.
  - This enriches the agent's knowledge base for improved context and responsiveness in future interactions.
- Facilitates the integration of external research into the agent's operational memory.
