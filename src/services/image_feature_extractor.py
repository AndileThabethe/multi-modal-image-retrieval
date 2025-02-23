import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from pymongo import MongoClient
import io

from sentence_transformers import SentenceTransformer, util

class ImageProcessor:
    def read_images_from_folder(folder_path):
        """
        Reads images from the specified folder and returns them as a list of image objects.

        Args:
            folder_path (str): The path to the folder containing the images.

        Returns:
            list: A list of image objects read from the folder.
        """
        print(f"Reading images from folder: {folder_path}")
        images = []
        try: 
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): 
                    img_path = os.path.join(folder_path, filename) 
                    try:  
                        img = Image.open(img_path)
                        images.append(img)
                        print(img)
                    except Exception as e:
                        print(f"Error opening image {filename}: {e}")
        except FileNotFoundError:
            print(f"Error: Folder not found at {folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return images

    def extract_features(images):
        """
        Extracts features from a list of image objects using the ResNet50 model.

        Args:
            images (list): A list of image objects.

        Returns:
            np.ndarray: A numpy array of extracted features.
        """
        print("Extracting features from images")
        model = ResNet50(weights='imagenet')
        features = []
        for img in images:
            try:
                img = img.resize((224, 224))
                img = image.img_to_array(img)
                img_arr = np.expand_dims(img.copy(), axis=0)
                img_arr = preprocess_input(img_arr)
                preds = model.predict(img_arr)
                preds = decode_predictions(preds, top=10)
                feature = " ".join([item[1] for item in preds[0]])
                features.append(feature)
            except Exception as e:
                print(f"Error processing image: {e}")
        return features

    def store_features_in_db(images, features, db_name='image_features_db', collection_name='features'):
        """
        Stores image features in a MongoDB database.
        Args:
            images (list): A list of PIL Image objects to be stored.
            features (list): A list of features corresponding to each image.
            db_name (str, optional): The name of the database. Defaults to 'image_features_db'.
            collection_name (str, optional): The name of the collection. Defaults to 'features'.
        """
        print(f"Storing features in database: {db_name}, collection: {collection_name}")
        client = MongoClient('localhost', 27017)
        db = client[db_name]
        collection = db[collection_name]

        model = SentenceTransformer('all-mpnet-base-v2')

        for img, feature in zip(images, features):
            i = 1
            try:
                print("Document: ", i)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                print("Image: ", img_byte_arr)
                print("Feature: ", feature)
                embeddings = model.encode(feature, convert_to_tensor=True).tolist()
                document = {
                    'image': img_byte_arr,
                    'feature': feature,
                    'embeddings': embeddings
                }
                i += 1
                collection.insert_one(document)
            except Exception as e:
                print(f"Error storing image and feature in database: {e}")