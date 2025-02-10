from imageprocessor import ImageProcessor as image_processor
from textqueryprocessor import TextQueryProcessor as text_processor
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from PIL import Image
import io, base64
import matplotlib.pyplot as plt


vectorizer = TfidfVectorizer()

query = "pigeon"
text_features = text_processor.preprocessing_query(query)
print(f"Text features shape: {text_features.shape}")

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

if text_features.shape[1] < image_features.shape[1]:
    padding = np.zeros((text_features.shape[0], image_features.shape[1] - text_features.shape[1]))
    text_features = np.hstack((text_features, padding))
elif image_features.shape[1] < text_features.shape[1]:
    padding = np.zeros((image_features.shape[0], text_features.shape[1] - image_features.shape[1]))
    image_features = np.hstack((image_features, padding))

similarities = cosine_similarity(text_features, image_features)

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

                                