import ollama
import mmap
import numpy as np

# --- Load dataset fast ---
dataset = []
with open('cat-facts.txt', 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for line in iter(mm.readline, b''):
            dataset.append(line.rstrip(b'\r\n').decode('utf-8', errors='replace'))
print(f'Loaded {len(dataset)} entries')

# --- Define models ---
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL  = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# --- Batch embed (single call instead of per line) ---
resp = ollama.embed(model=EMBEDDING_MODEL, input=dataset)
embeddings = resp['embeddings']  # list of embeddings, one per chunk
print(f'Generated {len(embeddings)} embeddings')

# --- Build NumPy arrays for fast retrieval ---
vecs  = np.array(embeddings, dtype=np.float32)
norms = np.linalg.norm(vecs, axis=1)

def retrieve(query, top_n=3):
    q = np.array(ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0], dtype=np.float32)
    qn = np.linalg.norm(q)
    sims = (vecs @ q) / (norms * (qn + 1e-12))
    top_idx = np.argpartition(-sims, top_n)[:top_n]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(dataset[i], float(sims[i])) for i in top_idx]

# --- Chatbot loop ---
while True:
    print()
    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query)

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n" +
        "\n".join([f' - {chunk}' for chunk, _ in retrieved_knowledge])
    )

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user',   'content': input_query},
        ],
        stream=True,
    )

    print('Chatbot response:')
    for part in stream:
        print(part['message']['content'], end='', flush=True)
