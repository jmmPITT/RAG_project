import getpass
import os

# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

file_path = "/workspaces/RAG_project/data/scientific_papers/PredictionAndMemory.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


print(all_splits[15])


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_splits)

results = vector_store.similarity_search(
    "What are opposing excitatory-inhibitory eﬀects mapped onto?"
)

print(results[0])


# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What are opposing excitatory-inhibitory eﬀects mapped onto?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)


embedding = embeddings.embed_query("What are opposing excitatory-inhibitory eﬀects mapped onto?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])