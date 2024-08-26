from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from agent import Agent, AgentConfig
from models import get_openai_chat, get_openai_embedding
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from python.helpers import extract_tools

load_dotenv()
from python.tools import memory_tool, knowledge_tool, online_knowledge_tool
import asyncio
import httpx
import uuid
from python.tools.home_assistant_tool import HomeAssistantTool

load_dotenv()

app = FastAPI()

# Initialize Agent
openai_api_key = os.getenv("OPENAI_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

chat_model = get_openai_chat(model_name="gpt-4o", api_key=openai_api_key)
utility_model = get_openai_chat(model_name="gpt-4o-mini", api_key=openai_api_key)
embedding_model = get_openai_embedding(model_name="text-embedding-3-small", api_key=openai_api_key)

config = AgentConfig(
    chat_model=chat_model,
    utility_model=utility_model,
    embeddings_model=embedding_model,
)

agents = {}

class AgentRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    device_id: str | None = None
    updates: bool = False
    timeout: int = 600  # Default to 10 minutes (600 seconds)

class MemoryRequest(BaseModel):
    text: str

class RecallRequest(BaseModel):
    prompt: str
    count: int = 5
    threshold: float = 0.1

class ResearchRequest(BaseModel):
    prompt: str = ""

async def initial_agent_setup(agent_id: str, prompt: str, conversation_id: str | None, device_id: str | None):
    agent = agents[agent_id]
    agent.append_message(prompt, human=True)
    
    if conversation_id and device_id:
        await send_conversation_response(conversation_id, device_id, "Agent started", original_prompt=prompt, is_final=False)

async def run_agent_task(agent_id: str, prompt: str, conversation_id: str | None, device_id: str | None, updates: bool, timeout: int):
    agent = agents[agent_id]
    
    try:
        # Perform initial setup and send "Agent started" message
        await initial_agent_setup(agent_id, prompt, conversation_id, device_id)
        
        async def update_callback(message: str):
            if updates and conversation_id and device_id:
                await send_conversation_response(conversation_id, device_id, message, original_prompt=prompt, is_final=False)
            await asyncio.to_thread(log_agent_response, agent_id, prompt, message, is_final=False)
        
        try:
            response = await asyncio.wait_for(agent.message_loop(prompt, update_callback=update_callback), timeout=timeout)
        except asyncio.TimeoutError:
            response = f"Agent task timed out after {timeout} seconds."
        
        # Log and send the final response
        await asyncio.to_thread(log_agent_response, agent_id, prompt, response, is_final=True)
        if conversation_id and device_id:
            await send_conversation_response(conversation_id, device_id, response, original_prompt=prompt, is_final=True)
    finally:
        # Clean up the agent
        del agents[agent_id]

def log_agent_response(agent_id: str, prompt: str, response: str, is_final: bool = False):
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"agent_{agent_id}.log")
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if is_final:
        logging.info(f"Final Response: {response}")
    else:
        logging.info(f"Interim Update: {response}")
        
@app.get("/run_agent_async")
async def run_agent_async(background_tasks: BackgroundTasks, prompt: str, conversation_id: str | None = None, device_id: str | None = None, updates: bool = False, timeout: int = 600):
    agent_id = str(uuid.uuid4())
    agents[agent_id] = Agent(number=len(agents), config=config)
    
    await asyncio.to_thread(log_agent_response, agent_id, prompt, "Agent started", is_final=False)
    
    asyncio.create_task(run_agent_task(agent_id, prompt, conversation_id, device_id, updates, timeout))
    
    return {"result": f"Agent {agent_id} started on the task"}

@app.get("/run_agent")
async def run_agent(prompt: str, conversation_id: str | None = None, device_id: str | None = None, updates: bool = False, timeout: int = 600):
    agent_id = str(uuid.uuid4())
    agent = Agent(number=len(agents), config=config)
    agents[agent_id] = agent
    
    try:
        await asyncio.to_thread(log_agent_response, agent_id, prompt, "Agent started", is_final=False)
        
        agent.append_message(prompt, human=True)
        
        try:
            response = await asyncio.wait_for(agent.message_loop(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            response = f"Agent task timed out after {timeout} seconds."
        
        await asyncio.to_thread(log_agent_response, agent_id, prompt, response, is_final=True)
        
        if conversation_id and device_id:
            await send_conversation_response(conversation_id, device_id, response, original_prompt=prompt, is_final=True)
        
        return {"result": response}
    finally:
        del agents[agent_id]

@app.get("/remember")
async def remember(text: str):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.save, agent, text)
    return {"result": result}

@app.get("/forget")
async def forget(prompt: str, count: int = 5, threshold: float = 0.1):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.forget, agent, prompt)
    return {"result": result}

@app.get("/recall")
async def recall(prompt: str, count: int = 5, threshold: float = 0.1):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.search, agent, prompt, count, threshold)
    return {"result": result}

@app.get("/research")
async def research(prompt: str = ""):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for research")
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    tool = knowledge_tool.Knowledge(agent=agent, name="knowledge", args={}, message="")
    response = await tool.execute(prompt=prompt)
    return {"result": response.message}

@app.get("/perplexity_search")
async def perplexity_search(prompt: str = ""):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    tool = online_knowledge_tool.OnlineKnowledge(agent=agent, name="online_knowledge", args={"prompt": prompt}, message="")
    response = await tool.execute()
    return {"result": response.message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
