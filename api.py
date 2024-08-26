from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from agent import Agent, AgentConfig
from models import get_openai_chat, get_openai_embedding
import os
from dotenv import load_dotenv
import logging
import asyncio
import uuid
from python.tools import memory_tool, knowledge_tool, online_knowledge_tool

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
    response_timeout_seconds=180,  # 3 minutes for individual agent response
)

agents = {}

class AgentRequest(BaseModel):
    prompt: str
    timeout: int | None = None

class MemoryRequest(BaseModel):
    text: str

class RecallRequest(BaseModel):
    prompt: str
    count: int = 5
    threshold: float = 0.1

class ResearchRequest(BaseModel):
    prompt: str

def log_agent_response(agent_id: str, prompt: str, response: str, is_final: bool = False):
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"agent_{agent_id}.log")
    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    if is_final:
        logging.info(f"Final Response: {response}")
    else:
        logging.info(f"Interim Update: {response}")

@app.post("/run_agent_async")
async def run_agent_async(request: AgentRequest, background_tasks: BackgroundTasks):
    agent_id = str(uuid.uuid4())
    agent = Agent(number=len(agents), config=config)
    agents[agent_id] = agent
    
    timeout = request.timeout or 900  # 15 minutes default for async runs

    background_tasks.add_task(run_agent_task, agent_id, request.prompt, timeout)
    
    return {"result": f"Agent {agent_id} started on the task"}

async def run_agent_task(agent_id: str, prompt: str, timeout: int):
    agent = agents[agent_id]
    
    try:
        await asyncio.to_thread(log_agent_response, agent_id, prompt, "Agent started", is_final=False)
        
        agent.append_message(prompt, human=True)
        
        try:
            response = await asyncio.wait_for(agent.message_loop(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            response = f"Agent task timed out after {timeout} seconds."
        
        await asyncio.to_thread(log_agent_response, agent_id, prompt, response, is_final=True)
        
    finally:
        del agents[agent_id]

@app.post("/run_agent")
async def run_agent(request: AgentRequest):
    agent_id = str(uuid.uuid4())
    agent = Agent(number=len(agents), config=config)
    agents[agent_id] = agent
    
    timeout = request.timeout or 180  # 3 minutes default for non-async runs
    
    try:
        await asyncio.to_thread(log_agent_response, agent_id, request.prompt, "Agent started", is_final=False)
        
        agent.append_message(request.prompt, human=True)
        
        try:
            response = await asyncio.wait_for(agent.message_loop(request.prompt), timeout=timeout)
        except asyncio.TimeoutError:
            response = f"Agent task timed out after {timeout} seconds."
        
        await asyncio.to_thread(log_agent_response, agent_id, request.prompt, response, is_final=True)
        
        return {"result": response}
    finally:
        del agents[agent_id]

@app.post("/remember")
async def remember(request: MemoryRequest):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.save, agent, request.text)
    return {"result": result}

@app.post("/forget")
async def forget(request: RecallRequest):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.forget, agent, request.prompt)
    return {"result": result}

@app.post("/recall")
async def recall(request: RecallRequest):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    result = await asyncio.to_thread(memory_tool.search, agent, request.prompt, request.count, request.threshold)
    return {"result": result}

@app.post("/research")
async def research(request: ResearchRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required for research")
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    tool = knowledge_tool.Knowledge(agent=agent, name="knowledge", args={}, message="")
    response = await tool.execute(prompt=request.prompt)
    return {"result": response.message}

@app.post("/perplexity_search")
async def perplexity_search(request: ResearchRequest):
    agent = next(iter(agents.values())) if agents else Agent(number=0, config=config)
    tool = online_knowledge_tool.OnlineKnowledge(agent=agent, name="online_knowledge", args={"prompt": request.prompt}, message="")
    response = await tool.execute()
    return {"result": response.message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
