import torch
if __name__!="__main__":
    from services.text_feature_extractor import TextQueryProcessor as text_processor
else:
    from text_feature_extractor import TextQueryProcessor as text_processor
from pymongo import MongoClient
from PIL import Image
import numpy as np
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

def data_query(text_prompt, K=5):
    """
    Queries the database for images similar to the given text prompt.
    Args:
        text_prompt (str): The text prompt to query the database with.
        K (int, optional): The number of top similar images to retrieve. Defaults to 5.
    Returns:
        list: A list of base64 encoded images sorted by similarity in descending order.
    Raises:
        Warning: If any image in the database is missing embeddings or image data.
        Exception: If there is an error decoding an image.
    Notes:
        - The function retrieves image embeddings from the database and computes their similarity to the query embedding.
        - It uses a pre-trained model to encode the text prompt into an embedding.
        - The function returns the top K most similar images in base64 format.
    """
    query = f"{text_prompt}"

    image_cursor = db["features"].find({}, {"embeddings": 1, "_id": 1})  
    all_embeddings = []
    image_ids = []

    for image in image_cursor: 
        embedding = image.get("embeddings")
        if embedding is None:
            print(f"Warning: Missing embedding for image ID: {image['_id']}", sys.stderr)
            continue 
        all_embeddings.append(np.array(embedding))
        image_ids.append(image['_id'])

    if not all_embeddings: 
        print("Warning: No embeddings found in the database.", sys.stderr)
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    embeddings_tensor = torch.tensor(np.stack(all_embeddings), dtype=torch.float32).to(device) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding = model.encode(query, convert_to_tensor=True).to(device).float() 
    
    similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)
    top_k_indices = torch.topk(similarities, K)[1][0].cpu().numpy() 

    neighbor_ids = [image_ids[i] for i in top_k_indices] 
    image_data = list(collection.find({"_id": {"$in": neighbor_ids}}))

    images = []
    for image_doc in image_data: 
        image_bytes = image_doc.get("image")
        if image_bytes is None:
            print(f"Warning: Missing image data for ID: {image_doc['_id']}", sys.stderr)
            continue

        try:
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image_to_base64(image))
        except Exception as e:
            print(f"Error decoding image: {e}", sys.stderr)

    return images[::-1]

if __name__ == "__main__":
    x = data_query("dog")
    print(x)    
