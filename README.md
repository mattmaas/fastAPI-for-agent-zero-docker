## Fork Overview

This fork adds an asynchronous FastAPI to the Agent Zero framework and also includes a Dockerfile for to easily run the original projecct in a self contained container. Key changes include:

- **FastAPI Integration**: The application now uses FastAPI to expose its functionalities as RESTful endpoints, allowing for easy interaction and integration with other services.
- **Agent Execution Methods**: Two methods for running agents are provided:
  - `run_agent`: Executes an agent synchronously, blocking until the task is complete.
  - `run_agent_async`: Initiates an agent task asynchronously, allowing other operations to continue while the agent works.
  - `research` - /Reserach endpoint now uses Pereplexica and SearXNG open source metadata search engine to generate sources and download FULL contents of up to 15 sources on a topic and summarize them and perplexica results using o1-mini from openai into an executive summary and saves it to the systems memories to "become an expert" on a subject. No longer uses duckduckgo. See the next section for more details.
  - `research_lite` - /Research_lite endpoint does not get the full contents of the Perplexica sources. Also saves to vectorDb agent memory.
  - `import-research` - /import-research imports knowledge and data stored in .txt documents into the agents' vectorDb memories.
  - `perplexity_search` - Does a quick perplexity request and also saves the response to vectorDb memory.
- **Asynchronous Enhancements**: Many operations have been refactored to be asynchronous, improving the responsiveness and scalability of the application.
- **Dockerization**: The application is designed to run inside a Docker container, making it easy to deploy as a microservice. The Dockerfile has been updated to reflect these changes.
- **Note on Docker**: Regular Dockerization features have been disabled in the code since the entire application is intended to run within a Docker container.

# Further API details

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

## /perplexity_search

The `/perplexity_search` endpoint utilizes the `perplexity_search` function defined in `python/helpers/perplexity_search.py`. It performs searches using the **Perplexity** API and returns concise answers based on the provided prompt.

- **Perplexity** is an AI-powered search engine that provides summarized information and direct answers from the web.
- Communicates with the Perplexity API to fetch real-time information.
- Handles messages, extracts responses, and logs interactions.
- Saves the search results to the memory bank for future use and prompting.
  - This allows the agent to reference Perplexity's insights in subsequent tasks.
- Useful for obtaining quick, succinct answers to specific queries without extensive processing.

## /import-research

The `/import-research` endpoint is handled by the `ResearchImport` tool defined in `python/tools/research_import_tool.py`. It allows importing existing research documents into the agent's memory bank.

- Scans a specified directory for `.txt` research files.
- Reads and processes each research document.
- Uses OpenAI models to generate concise memory prompts that capture the essence of each document.
- Saves both the original content and the generated prompts to the agent's memory bank.
  - This enriches the agent's knowledge base for improved context and responsiveness in future interactions.
- Facilitates the integration of external research into the agent's operational memory.

# Agent Zero

[![Join our Skool Community](https://img.shields.io/badge/Skool-Join%20our%20Community-4A90E2?style=for-the-badge&logo=skool&logoColor=white)](https://www.skool.com/agent-zero) [![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/B8KZKNsPpj) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AgentZeroFW) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jan-tomasek/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/JanTomasekDev)


[![Intro Video](/docs/res/intro_vid.jpg)](https://www.youtube.com/watch?v=C9n8zFpaV3I)

**Personal and organic AI framework**
- Agent Zero is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
- Agent Zero is fully transparent, readable, comprehensible, customizable and interactive.
- Agent Zero uses the computer as a tool to accomplish its (your) tasks.

## Now with UI:
![UI prototype](/docs/res/ui_screen1.png)

## Key concepts
1. **General-purpose assistant**
- Agent Zero is not pre-programmed for specific tasks (but can be). It is meant to be a general-purpose personal assistant. Give it a task, and it will gather information, execute commands and code, cooperate with other agent instances, and do its best to accomplish it.
- It has a persistent memory, allowing it to memorize previous solutions, code, facts, instructions, etc., to solve tasks faster and more reliably in the future.

2. **Computer as a tool**
- Agent Zero uses the operating system as a tool to accomplish its tasks. It has no single-purpose tools pre-programmed. Instead, it can write its own code and use the terminal to create and use its own tools as needed.
- The only default tools in its arsenal are online search, memory features, communication (with the user and other agents), and code/terminal execution. Everything else is created by the agent itself or can be extended by the user.
- Tool usage functionality has been developed from scratch to be the most compatible and reliable, even with very small models.

3. **Multi-agent cooperation**
- Every agent has a superior agent giving it tasks and instructions. Every agent then reports back to its superior.
- In the case of the first agent in the chain (Agent 0), the superior is the human user; the agent sees no difference.
- Every agent can create its subordinate agent to help break down and solve subtasks. This helps all agents keep their context clean and focused.

4. **Completely customizable and extensible**
- Almost nothing in this framework is hard-coded. Nothing is hidden. Everything can be extended or changed by the user.
- The whole behavior is defined by a system prompt in the **prompts/default/agent.system.md** file. Change this prompt and change the framework dramatically.
- The framework does not guide or limit the agent in any way. There are no hard-coded rails that agents have to follow.
- Every prompt, every small message template sent to the agent in its communication loop, can be found in the **prompts/** folder and changed.
- Every default tool can be found in the **python/tools/** folder and changed or copied to create new predefined tools.
- Of course, it is open-source (except for some tools like Perplexity, but that will be replaced with an open-source alternative as well in the future).

5. **Communication is key**
- Give your agent a proper system prompt and instructions, and it can do miracles.
- Agents can communicate with their superiors and subordinates, asking questions, giving instructions, and providing guidance. Instruct your agents in the system prompt on how to communicate effectively.
- The terminal interface is real-time streamed and interactive. You can stop and intervene at any point. If you see your agent heading in the wrong direction, just stop and tell it right away.
- There is a lot of freedom in this framework. You can instruct your agents to regularly report back to superiors asking for permission to continue. You can instruct them to use point-scoring systems when deciding when to delegate subtasks. Superiors can double-check subordinates' results and dispute. The possibilities are endless.

![Agent Zero](/docs/res/splash_wide.png)

## Nice features to have
- Output is very clean, colorful, readable and interactive; nothing is hidden.
- The same colorful output you see in the terminal is automatically saved to HTML file in **logs/** folder for every session.
- Agent output is streamed in real-time, allowing the user to read along and intervene at any time.
- No coding is required, only prompting and communication skills.
- With a solid system prompt, the framework is reliable even with small models, including precise tool usage.

## Keep in mind
1. **Agent Zero can be dangerous!**
With proper instruction, Agent Zero is capable of many things, even potentially dangerous to your computer, data, or accounts. Always run Agent Zero in an isolated environment (like the built in docker container) and be careful what you wish for.

2. **Agent Zero is not pre-programmed; it is prompt-based.**
The whole framework contains only a minimal amount of code and does not guide the agent in any way.
Everything lies in the system prompt in the **prompts/** folder. Here you can rewrite the whole framework behavior to your needs.
If your agent fails to communicate properly, use tools, reason, use memory, find answers - just instruct it better.

3. **If you cannot provide the ideal environment, let your agent know.**
Agent Zero is made to be used in an isolated virtual environment (for safety) with some tools preinstalled and configured.
If you cannot provide all the necessary conditions or API keys, just change the system prompt and tell your agent what operating system and tools are at its disposal. Nothing is hard-coded; if you do not tell your agent about a certain tool, it will not know about it and will not try to use it.


[![David Ondrej video](/docs/res/david_vid.jpg)](https://www.youtube.com/watch?v=_Pionjv4hGc)

## Known problems
1. The system prompt sucks. You can do better. If you do, help me please :)
2. The communication between agent and terminal in docker container via SSH can sometimes break and stop producing outputs. Sometimes it is because the agent runs something like "server.serve_forever()" which causes the terminal to hang, sometimes a random error can occur. Restarting the agent and/or the docker container helps.
3. The agent can break his operating system. Sometimes the agent can deactivate virtual environment, uninstall packages, change config etc. Again, removing the docker container and cleaning up the **work_dir/** is enough to fix that.

## Ideal environment
- **Docker container**: The perfect environment to run Agent Zero is the built-in docker container. The agent can download the image **frdel/agent-zero-exe** on its own and start the container, you only need to have docker running (like the Docker Desktop application).
- **Python**: Python has to be installed on the system to run the framework.
- **Internet access**: The agent will need internet access to use its online knowledge tool and execute commands and scripts requiring a connection. If you do not need your agent to be online, you can alter its prompts in the **prompts/** folder and make it fully local.

![Time example](/docs/res/time_example.jpg)

## Setup

A detailed setup guide with a video can be found here: [/docs/installation](https://github.com/frdel/agent-zero/tree/main/docs/installation). Scroll down to see the readme file.

>  **Changes to launch files since v0.6:**  
> main.py file has been replaced with run_ui.py (webui) and run_cli.py (terminal) launch files.
> configuration has been moved to initialize.py for both webui and terminal launch files.
