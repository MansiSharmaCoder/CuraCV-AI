from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np


anime_data = [
    {"name": "Naruto Uzumaki", "desc": "A ninja with a dream to become Hokage"},
    {"name": "Sasuke Uchiha",  "desc": "A vengeful ninja from the Uchiha clan"},
    {"name": "Luffy",          "desc": "Apirate with a rubber body and a big dream"},
    {"name": "Goku",           "desc": "Saiyan warrior who protects Earth"},
    {"name": "Light Yagami",   "desc": "A genius who gets a death note"}
]



model = SentenceTransformer('all-MiniLM-L6-v2')


descriptions = [char["desc"] for char in anime_data]


vectors= model.encode(descriptions)

index = faiss.IndexFlatL2(384)

index.add(np.array(vectors))

query = "who is the strongest ninja ?"

query_vector = model.encode([query])

D,I=index.search(np.array(query_vector),k=2)

for idx in I[0]:
    print(f"your answer is {anime_data[idx]["name"]}")

    print(D)
    print(I)