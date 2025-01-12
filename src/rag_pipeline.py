# rag_pipeline.py

import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# If you want to reuse the pdf_extractor from your project:
from pdf_extractor import extract_text_from_pdf

# 1. LOAD TEXT FROM PDF
def load_pdf_text():
    pdf_path = os.path.join("data", "scientific_papers", "FEPmadeSimple.pdf")
    return extract_text_from_pdf(pdf_path)

# 2. SPLIT TEXT INTO CHUNKS (if needed)
def chunk_text(text, chunk_size=512, overlap=50):
    """
    Splits text into chunks for embedding.
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap  # overlap for better context
        if start < 0:
            start = 0
    
    return chunks

# 3. EMBED TEXT USING A MODEL (e.g., sentence-transformers)
def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def compute_embeddings(tokenizer, model, texts):
    """
    Compute embeddings for a list of texts using a huggingface transformer model.
    """
    # Convert to tokens
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Pool the [CLS] token for simplicity
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

# 4. BUILD A VECTOR STORE (using FAISS)
def build_faiss_index(embeddings):
    d = embeddings.shape[1]  # dimension of embedding
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# 5. RETRIEVE TOP-K CHUNKS
def retrieve_similar_chunks(query_embedding, index, embeddings, top_k=3):
    """
    Retrieve the top_k most similar chunks for a given query embedding.
    """
    # FAISS expects shape [n, d]
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

# 6. GENERATE ANSWER WITH LLM (for the sake of example, we’ll just load a simple pipeline)
def generate_answer(context, query, llm_model="google/flan-t5-small"):
    """
    Uses a generative model to produce an answer from the provided context & query.
    """
    generator = pipeline("text2text-generation", model=llm_model)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]["generated_text"]

# MAIN RAG LOGIC
if __name__ == "__main__":
    # Step A: Load PDF text
    pdf_text = load_pdf_text()
    
    # Step B: Split PDF text into chunks
    text_chunks = chunk_text(pdf_text)
    
    # Step C: Load embedding model & compute embeddings
    tokenizer, emb_model = load_embedding_model()
    doc_embeddings = compute_embeddings(tokenizer, emb_model, text_chunks)

    # Step D: Build FAISS index
    faiss_index = build_faiss_index(doc_embeddings)
    
    # Step E: For a user query, embed the query
    user_query = "What is the main idea of FEPmadeSimple?"
    query_emb = compute_embeddings(tokenizer, emb_model, [user_query])[0]  # Single query

    # Step F: Retrieve top-k chunks
    top_indices, distances = retrieve_similar_chunks(query_emb, faiss_index, doc_embeddings, top_k=3)
    
    # Gather top chunks
    retrieved_context = "\n".join([text_chunks[i] for i in top_indices])
    
    # Step G: Generate answer
    answer = generate_answer(retrieved_context, user_query)
    print("Answer:", answer)
