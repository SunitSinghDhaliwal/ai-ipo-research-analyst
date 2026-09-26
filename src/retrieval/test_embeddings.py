from embeddings import get_embedding_model


model = get_embedding_model()

text = "The company reported an increase in revenue."

vector = model.embed_query(text)

print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])