from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt_tab')

# Load and tokenize the text
def load_and_tokenize(file_path):
    text=file_path[1].values()
    return sent_tokenize(text)


# Combine sentences with their neighbors
def combine_sentences(sentences, buffer=1):
    combined = []
    for i in range(len(sentences)):
        start = max(0, i - buffer)
        end = min(len(sentences), i + buffer + 1)
        combined.append(' '.join(sentences[start:end]))
    return combined

# Calculate cosine distances between embeddings
def calc_cosine_distances(embeddings):
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        distances.append(1 - sim)
    return distances

# Find breakpoints based on distance threshold
def find_breakpoints(distances, percentile=95):
    threshold = np.percentile(distances, percentile)
    return [i for i, d in enumerate(distances) if d > threshold]

# Create chunks based on breakpoints
def create_chunks(sentences, breakpoints):
    chunks = []
    start = 0
    for bp in breakpoints:
        chunks.append(' '.join(sentences[start:bp + 1]))
        start = bp + 1
    chunks.append(' '.join(sentences[start:]))
    return chunks

# Merge small chunks with their most similar neighbor
def merge_small_chunks(chunks, embeddings, min_size=3):
    merged = [chunks[0]]
    merged_emb = [embeddings[0]]

    for i in range(1, len(chunks) - 1):
        if len(chunks[i].split('. ')) < min_size:
            prev_sim = cosine_similarity([embeddings[i]], [merged_emb[-1]])[0][0]
            next_sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]

            if prev_sim > next_sim:
                merged[-1] += ' ' + chunks[i]
                merged_emb[-1] = (merged_emb[-1] + embeddings[i]) / 2
            else:
                chunks[i + 1] = chunks[i] + ' ' + chunks[i + 1]
                embeddings[i + 1] = (embeddings[i] + embeddings[i + 1]) / 2
        else:
            merged.append(chunks[i])
            merged_emb.append(embeddings[i])

    merged.append(chunks[-1])
    merged_emb.append(embeddings[-1])
    return merged, merged_emb

# Main process
def chunk_text(file_path):
    # Load the model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')

    # Process the text
    sentences = load_and_tokenize(file_path)
    combined = combine_sentences(sentences)
    embeddings = model.encode(combined)

    # Find breakpoints and create initial chunks
    distances = calc_cosine_distances(embeddings)
    breakpoints = find_breakpoints(distances)
    chunks = create_chunks(sentences, breakpoints)

    # Merge small chunks
    chunk_embeddings = model.encode(chunks)
    final_chunks, _ = merge_small_chunks(chunks, chunk_embeddings)

    return final_chunks

if __name__ == "__main__":
    reader =PdfReader("HOML.pdf")
    pages=[]
    for i ,page in enumerate(reader.pages):
        pages.append({
            "page":i+1,
            "text":page.extract_text(),
        })
    result = chunk_text(pages)
    print(f"Number of chunks: {len(result)}")
    print("First chunk:", result[0][:100] + "...")
