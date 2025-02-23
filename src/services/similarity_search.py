import torch
from services.text_feature_extractor import TextQueryProcessor as text_processor
from pymongo import MongoClient
from PIL import Image
import base64
from io import BytesIO
import io
import sys

from sentence_transformers import SentenceTransformer, util
from datetime import datetime

model = SentenceTransformer('all-mpnet-base-v2')

client = MongoClient("mongodb://localhost:27017/")
db = client["image_features_db"]
collection = db["features"]

def semantic_similarity(text, embedding):
    """
    Computes the semantic similarity between two texts using pre-trained embeddings.
    Args:
        text1 (str): The first text input.
        text2 (str): The second text input.
    Returns:
        float: The cosine similarity score between the embeddings of the two texts.
    """
    embeddings1 = model.encode(text, convert_to_tensor=True)
    embeddings2 = torch.tensor(embedding)

    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()

def image_to_base64(image):
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image (PIL.Image.Image): The image to be converted.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def data_query(text_prompt):
    """
    Queries the database for images similar to the given text prompt.
    Args:
        text_prompt (str): The text prompt to query for similar images.
    Returns:
        list: A list of base64 encoded images that are most similar to the text prompt.
    """
    query = f"{text_prompt}"
    text_features = text_processor.preprocessing_text(query)

    similarity_dict = {}
    image_collection = db["features"].find()

    print("Calculating semantic similarity scores...", sys.stderr)
    print(f"Start Timestamp: {datetime.now()}", sys.stderr)
    for image in image_collection:
        similarity = semantic_similarity(query, image['embeddings'])
        similarity_dict[image['_id']] = similarity
    print(f"End Timestamp: {datetime.now()}", sys.stderr)

    K = 5
    sorted_similarities = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
    nearest_neighbors = sorted_similarities[:K]
    image_data = [collection.find_one({"_id": neighbor[0]}) for neighbor in nearest_neighbors]
    images = [Image.open(io.BytesIO(image_data['image'])) for image_data in image_data]


    im = [image_to_base64(image) for image in images]
    return im
