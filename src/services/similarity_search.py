# from image_feature_extractor import ImageProcessor as image_processor
from services.text_feature_extractor import TextQueryProcessor as text_processor
from pymongo import MongoClient
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import base64
from PIL import Image
import base64
from io import BytesIO
import io
# import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
def semantic_similarity(text1, text2):
    # global SentenceTransformer
    
    # from sentence_transformers import SentenceTransformer, util
# import numpy as np
    # Encode the sentences
    
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    # Compute the cosine similarity
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# def mongo_db():
client = MongoClient("mongodb://localhost:27017/")
db = client["image_features_db"]
collection = db["features"]
# return db, collection

# db, collection = mongo_db()
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def data_query(text_prompt):
    # db, collection = mongo_db()
    query = f"{text_prompt}"
    text_features = text_processor.preprocessing_text(query)
    # print(f"Text features shape: {text_features.len}")

    similarity_dict = {}
    image_collection = db["features"].find()

    for image in image_collection:
        similarity = semantic_similarity(query, image['feature'])
        similarity_dict[image['_id']] = similarity

    K = 5
    sorted_similarities = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
    nearest_neighbors = sorted_similarities[:K]
    image_data = [collection.find_one({"_id": neighbor[0]}) for neighbor in nearest_neighbors]
    images = [Image.open(io.BytesIO(image_data['image'])) for image_data in image_data]


    im = [image_to_base64(image) for image in images]
    return im

# for neighbor in nearest_neighbors:
#     image_data = collection.find_one({"_id": neighbor[0]})
#     image = Image.open(io.BytesIO(image_data['image']))
#     plt.imshow(image)
#     plt.title(f"Image ID: {neighbor[0]}, Similarity: {neighbor[1]}")
#     plt.axis('off')
#     plt.show()

if __name__ == "__main__":
    text_prompt = "A beautiful sunset"
    x = data_query(text_prompt)
    print("Done")