import numpy as np
from sentence_transformers import SentenceTransformer, util  # noqa: E402


EMBEDDINGS_MODEL= 'all-MiniLM-L6-v2'

# not necessary - for learning purposes
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    # Calculate the similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)   
    return dot_product / (norm_vec1 * norm_vec2)

def calculate_cosine_distances(sentence1: np.ndarray, sentence2: np.ndarray) -> float:
    # Calculate cosine similarity
    similarity = cosine_similarity(sentence1, sentence2)

    return 1- similarity

def calculate_cosine_similarity(text1: str, text2: str) -> float: 
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = model.encode([text1, text2])
    #print(embeddings)
    #print(embeddings.shape)
    #similarity = cosine_similarity(embeddings[0], embeddings[1])
    #return similarity
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

if __name__ == "__main__": 
    print(calculate_cosine_similarity("hello jason", "hello new world")) 
    print(calculate_cosine_similarity("hello ", "hello new world"))