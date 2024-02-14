from sentence_transformers import SentenceTransformer

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sentences to encode
sentences = [
    "John is the boss",
    "Ansh will be the boss",
    "John is boss"
]

embeddings = model.encode(sentences)

def dot_product(a, b):
    
    return sum(a_i * b_i for a_i, b_i in zip(a, b))

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity = dot_product(embeddings[i], embeddings[j])
        print(f"Similarity between '{sentences[i]}' and '{sentences[j]}': {similarity:.4f}")
