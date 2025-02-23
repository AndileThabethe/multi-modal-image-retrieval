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

# def data_query(text_prompt):
#     """
#     Queries the database for images similar to the given text prompt.
#     Args:
#         text_prompt (str): The text prompt to query for similar images.
#     Returns:
#         list: A list of base64 encoded images that are most similar to the text prompt.
#     """
#     query = f"{text_prompt}"
#     text_features = text_processor.preprocessing_text(query)

#     similarity_dict = {}
#     image_collection = db["features"].find()

#     print("Calculating semantic similarity scores...", sys.stderr)
#     print(f"Start Timestamp: {datetime.now()}", sys.stderr)
#     for image in image_collection:
#         similarity = semantic_similarity(query, image['embeddings'])
#         similarity_dict[image['_id']] = similarity
#     print(f"End Timestamp: {datetime.now()}", sys.stderr)

#     K = 5
#     sorted_similarities = sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True)
#     nearest_neighbors = sorted_similarities[:K]
#     image_data = [collection.find_one({"_id": neighbor[0]}) for neighbor in nearest_neighbors]
#     images = [Image.open(io.BytesIO(image_data['image'])) for image_data in image_data]


#     im = [image_to_base64(image) for image in images]
#     return im

def data_query(text_prompt, K=5):
    query = f"{text_prompt}"

    print("Calculating semantic similarity scores...", sys.stderr)
    print(f"Start Timestamp: {datetime.now()}", sys.stderr)

    # 1. Efficient Retrieval of Embeddings and IDs (optimized):
    image_cursor = db["features"].find({}, {"embeddings": 1, "_id": 1})  # Use cursor for large datasets

    all_embeddings = []
    image_ids = []

    for image in image_cursor: # Efficient iteration for large datasets
        embedding = image.get("embeddings") # Handle potential missing embeddings
        if embedding is None:
            print(f"Warning: Missing embedding for image ID: {image['_id']}", sys.stderr)
            continue # Skip image if embedding is missing

        all_embeddings.append(np.array(embedding))
        image_ids.append(image['_id'])

    if not all_embeddings: # Handle case where no embeddings are found
        print("Warning: No embeddings found in the database.", sys.stderr)
        return []

    # 2. Convert to Tensor (optimized):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device HERE
    embeddings_tensor = torch.tensor(np.stack(all_embeddings), dtype=torch.float32).to(device) # Combine stack and to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Encode Query (only once) and move to device:
    query_embedding = model.encode(query, convert_to_tensor=True).to(device).float() # Combine operations

    # 4. Batch Similarity Calculation (optimized):
    similarities = util.pytorch_cos_sim(query_embedding, embeddings_tensor)

    # 5. Get Top K Indices (optimized):
    top_k_indices = torch.topk(similarities, K)[1][0].cpu().numpy()  # Get indices directly

    # 6. Efficient Retrieval of Image Data (using $in operator):
    neighbor_ids = [image_ids[i] for i in top_k_indices] # Use indices to get IDs
    image_data = list(collection.find({"_id": {"$in": neighbor_ids}}))

    print(f"End Timestamp: {datetime.now()}", sys.stderr)

    # 7. Open Images and Convert to Base64 (optimized):
    images = []
    for image_doc in image_data: # Handle missing images
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
