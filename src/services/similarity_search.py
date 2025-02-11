from image_feature_extractor import ImageProcessor as image_processor
from text_feature_extractor import TextQueryProcessor as text_processor
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from PIL import Image
import io, base64
import matplotlib.pyplot as plt

vectorizer = TfidfVectorizer()

query = "a time piece on a table"
text_features = text_processor.preprocessing_query(query)
print(f"Text features shape: {text_features.len}")

client = MongoClient("mongodb://localhost:27017/")
db = client["image_features_db"]
collection = db["features"]

image_collection = db["features"].find()
image_features = []
image_ids = []
for image in image_collection:
    image_features.append(image['feature'])
    image_ids.append(image['_id'])
image_features = np.array(image_features)
print(f"Image features shape: {image_features.shape}")

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

similarities = cosine_sim(text_features, image_features)

K = 5
nearest_neighbors_indices = np.argsort(similarities[0])[-K:][::-1]
nearest_neighbors = [(image_ids[idx], similarities[0][idx]) for idx in nearest_neighbors_indices]

for neighbor in nearest_neighbors:
    print(f"Image ID: {neighbor[0]}, Similarity: {neighbor[1]}")

nearest_neighbor_ids = [neighbor[0] for neighbor in nearest_neighbors]

images = []
for neighbor_id in nearest_neighbor_ids:
    image = collection.find_one({"_id": neighbor_id})
    image_data = image['image']
    
    img = Image.open(io.BytesIO(image_data))
    plt.imshow(img)
    plt.axis('off')  
    plt.show()

                                