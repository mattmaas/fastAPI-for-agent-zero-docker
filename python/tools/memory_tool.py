import re
import logging
from agent import Agent
from python.helpers.vector_db import VectorDB, Document
from python.helpers import files
import os, json
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from chromadb.errors import InvalidDimensionException

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO multiple DBs at once
db: VectorDB | None= None

class Memory(Tool):
    async def execute(self, **kwargs):
        logger.debug(f"Memory.execute called with kwargs: {kwargs}")
        result = ""
    
        try:
            if "query" in kwargs:
                threshold = float(kwargs.get("threshold", 0.1))
                count = int(kwargs.get("count", 5))
                logger.debug(f"Executing search with query: {kwargs['query']}, count: {count}, threshold: {threshold}")
                result = await search(self.agent, kwargs["query"], count, threshold)
            elif "memorize" in kwargs:
                logger.debug(f"Executing save with text: {kwargs['memorize']}")
                result = await save(self.agent, kwargs["memorize"])
            elif "forget" in kwargs:
                logger.debug(f"Executing forget with query: {kwargs['forget']}")
                result = await forget(self.agent, kwargs["forget"])
            elif "delete" in kwargs:
                logger.debug(f"Executing delete with ids: {kwargs['delete']}")
                result = await delete(self.agent, kwargs["delete"])
            else:
                logger.warning("No recognized operation in kwargs")
                return Response(message="No recognized operation", break_loop=False)
        except InvalidDimensionException as e:
            # hint about embedding change with existing database
            PrintStyle.hint("If you changed your embedding model, you will need to remove contents of /memory directory.")
            logger.error(f"InvalidDimensionException: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Memory.execute: {str(e)}")
            return Response(message=f"An error occurred: {str(e)}", break_loop=False)
    
        logger.debug(f"Memory.execute returning result: {result}")
        return Response(message=result, break_loop=False)
            
async def search(agent:Agent, query:str, count:int=5, threshold:float=0.1):
    initialize(agent)
    docs = await db.search_similarity_threshold(query,count,threshold) # type: ignore
    if not docs:  # Check if docs is empty (None or empty list)
        return files.read_file("./prompts/fw.memories_not_found.md", query=query)
    else: 
        return str(docs)

async def save(agent:Agent, text:str):
    initialize(agent)
    id = await db.insert_document(text) # type: ignore
    return files.read_file("./prompts/fw.memory_saved.md", memory_id=id)

async def delete(agent:Agent, ids_str:str):
    initialize(agent)
    ids = extract_guids(ids_str)
    deleted = await db.delete_documents_by_ids(ids) # type: ignore
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)    

async def forget(agent:Agent, query:str):
    initialize(agent)
    deleted = await db.delete_documents_by_query(query) # type: ignore
    return files.read_file("./prompts/fw.memories_deleted.md", memory_count=deleted)

def initialize(agent:Agent):
    global db
    if not db:
        dir = os.path.join("memory",agent.config.memory_subdir)
        db = VectorDB(embeddings_model=agent.config.embeddings_model, in_memory=False, cache_dir=dir)

def extract_guids(text):
    pattern = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
    return re.findall(pattern, text)
