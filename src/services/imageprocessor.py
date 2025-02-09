import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from pymongo import MongoClient
import io

## to do:
# add image preprocessing for enhanced feature extraction

class ImageProcessor:
    def read_images_from_folder(folder_path):
        print(f"Reading images from folder: {folder_path}")
        images = []
        try: 
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): 
                    img_path = os.path.join(folder_path, filename) 
                    try:  
                        img = Image.open(img_path)
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {filename}: {e}")
        except FileNotFoundError:
            print(f"Error: Folder not found at {folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return images


    def extract_features(images):
        print("Extracting features from images")
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        features = []
        for img in images:
            try:
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                feature = model.predict(img_array)
                features.append(feature.flatten())
            except Exception as e:
                print(f"Error processing image: {e}")
        return np.array(features)

    def store_features_in_db(images, features, db_name='image_features_db', collection_name='features'):
        print(f"Storing features in database: {db_name}, collection: {collection_name}")
        client = MongoClient('localhost', 27017)
        db = client[db_name]
        collection = db[collection_name]

        for img, feature in zip(images, features):
            try:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                document = {
                    'image': img_byte_arr,
                    'feature': feature.tolist()
                }
                collection.insert_one(document)
            except Exception as e:
                print(f"Error storing image and feature in database: {e}")