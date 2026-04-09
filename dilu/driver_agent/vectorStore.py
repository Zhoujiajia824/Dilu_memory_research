import os
import shutil

# 尝试导入 Chroma，如果不存在则报错
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        raise ImportError(
            "无法导入 Chroma。请安装依赖:\n"
            "  pip install langchain-community chromadb"
        )

# 简单的 Document 类，无需依赖 langchain
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

from dilu.driver_agent.model_provider import build_embedding_model, get_embedding_signature
from dilu.scenario.envScenario import EnvScenario


class DrivingMemory:

    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type == 'sce_encode':
            # 'sce_encode' is deprecated for now.
            raise ValueError("encode_type sce_encode is deprecated for now.")
        elif encode_type == 'sce_language':
            self.embedding = build_embedding_model()
            self.embedding_signature = get_embedding_signature()
            self.embedding_dimension = None
            db_path = os.path.join(
                './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
            self.db_path = db_path
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
            self._ensure_embedding_dimension_compatible()
        else:
            raise ValueError(
                "Unknown ENCODE_TYPE: should be sce_encode or sce_language")
        print("==========Loaded ",db_path," Memory, Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

    def _ensure_embedding_dimension_compatible(self):
        stored_items = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        embeddings = stored_items.get('embeddings') or []
        documents = stored_items.get('documents') or []
        metadatas = stored_items.get('metadatas') or []
        if not embeddings or not documents:
            return

        stored_dimension = len(embeddings[0])
        current_dimension = len(self.embedding.embed_query(documents[0]))
        self.embedding_dimension = current_dimension
        metadata_signature = self._extract_metadata_signature(metadatas)
        expected_signature = self._current_metadata_signature(current_dimension)
        if stored_dimension == current_dimension and metadata_signature == expected_signature:
            return

        reasons = []
        if stored_dimension != current_dimension:
            reasons.append(f"dimension stored={stored_dimension}, current={current_dimension}")
        if metadata_signature != expected_signature:
            reasons.append(
                "signature "
                f"stored={metadata_signature}, current={expected_signature}"
            )
        print(
            f"Incompatible memory detected in {self.db_path}: "
            + "; ".join(reasons)
            + ". Rebuilding memory vectors."
        )
        ids = stored_items['ids']
        rebuilt_metadatas = [
            self._merge_metadata(metadata, current_dimension)
            for metadata in metadatas
        ]
        shutil.rmtree(self.db_path, ignore_errors=True)
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.db_path
        )
        new_embeddings = self.embedding.embed_documents(documents)
        self.embedding_dimension = len(new_embeddings[0])
        self.scenario_memory._collection.add(
            embeddings=new_embeddings,
            metadatas=rebuilt_metadatas,
            documents=documents,
            ids=ids,
        )
        self.scenario_memory.persist()

    def _extract_metadata_signature(self, metadatas):
        if not metadatas:
            return {}
        first_metadata = metadatas[0] or {}
        return {
            "provider": first_metadata.get("provider"),
            "embed_model": first_metadata.get("embed_model"),
            "dimension": first_metadata.get("dimension"),
        }

    def _current_metadata_signature(self, dimension):
        return {
            "provider": self.embedding_signature["provider"],
            "embed_model": self.embedding_signature["embed_model"],
            "dimension": dimension,
        }

    def _get_embedding_dimension(self, sample_text):
        if self.embedding_dimension is None:
            self.embedding_dimension = len(self.embedding.embed_query(sample_text))
        return self.embedding_dimension

    def _merge_metadata(self, metadata, dimension, **extra_fields):
        merged = dict(metadata or {})
        merged.update(extra_fields)
        merged.update(self._current_metadata_signature(dimension))
        return merged

    def retriveMemory(self, driving_scenario: EnvScenario, frame_id: int, top_k: int = 5):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            query_scenario = driving_scenario.describe(frame_id)
            similarity_results = self.scenario_memory.similarity_search_with_score(
                query_scenario, k=top_k)
            fewshot_results = []
            for idx in range(0, len(similarity_results)):
                # print(f"similarity score: {similarity_results[idx][1]}")
                fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = ""):
        if self.encode_type == 'sce_encode':
            pass
        elif self.encode_type == 'sce_language':
            sce_descrip = sce_descrip.replace("'", '')
        # https://docs.trychroma.com/usage-guide#using-where-filters
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )
        # print("get_results: ", get_results)

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas=self._merge_metadata(
                    {
                        "human_question": human_question,
                        'LLM_response': response,
                        'action': action,
                        'comments': comments,
                    },
                    self._get_embedding_dimension(sce_descrip)
                )
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            embedding_dimension = self._get_embedding_dimension(sce_descrip)
            doc = Document(
                page_content=sce_descrip,
                metadata=self._merge_metadata(
                    {
                        "human_question": human_question,
                        'LLM_response': response,
                        'action': action,
                        'comments': comments,
                    },
                    embedding_dimension
                )
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")


if __name__ == "__main__":
    pass
