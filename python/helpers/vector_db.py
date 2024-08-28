from langchain.storage import InMemoryByteStore, LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma

from . import files
from langchain_core.documents import Document
import uuid


class VectorDB:

    def __init__(self, embeddings_model, in_memory=False, cache_dir="./cache"):
        print("Initializing VectorDB...")
        self.embeddings_model = embeddings_model

        em_cache = files.get_abs_path(cache_dir,"embeddings")
        db_cache = files.get_abs_path(cache_dir,"database")
        
        if in_memory:
            self.store = InMemoryByteStore()
        else:
            self.store = LocalFileStore(em_cache)


        #here we setup the embeddings model with the chosen cache storage
        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            embeddings_model, 
            self.store, 
            namespace=getattr(embeddings_model, 'model', getattr(embeddings_model, 'model_name', "default")) )


        self.db = Chroma(embedding_function=self.embedder,persist_directory=db_cache)
        
        
    def search_similarity(self, query, results=3):
        return self.db.similarity_search(query,results)
    
    async def search_similarity_threshold(self, query, results=3, threshold=0.5):
        return await self.db.asearch(query, search_type="similarity_score_threshold", k=results, score_threshold=threshold)

    def search_max_rel(self, query, results=3):
        return self.db.max_marginal_relevance_search(query,results)

    async def delete_documents_by_query(self, query:str, threshold=0.1):
        k = 100
        tot = 0
        while True:
            # Perform similarity search with score
            docs = await self.search_similarity_threshold(query, results=k, threshold=threshold)

            # Extract document IDs and filter based on score
            document_ids = [result.metadata["id"] for result in docs]
            

            # Delete documents with IDs over the threshold score
            if document_ids:
                await self.db.adelete(ids=document_ids)
                tot += len(document_ids)
                                    
            # If fewer than K document IDs, break the loop
            if len(document_ids) < k:
                break
        
        return tot

    async def delete_documents_by_ids(self, ids:list[str]):
        # pre = self.db.get(ids=ids)["ids"]
        await self.db.adelete(ids=ids)
        # post = self.db.get(ids=ids)["ids"]
        #TODO? compare pre and post
        return len(ids)
        
    async def insert_document(self, data):
        id = str(uuid.uuid4())
        await self.db.aadd_documents(documents=[ Document(data, metadata={"id": id}) ], ids=[id])
        
        return id
        


