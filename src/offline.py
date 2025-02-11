"""
This script extracts features from images and stores them in a MongoDB database.
The features are extracted using a pre-trained ResNet50 model. The images are read from a specified folder,
and the features are stored in the database for further use, such as image similarity search or other image processing tasks.
Modules:
    services.imageprocessor: Contains the ImageProcessor class for reading images, extracting features, and storing them in the database.
Usage:
    This script is used tp process images and push them to the database.
Functions:
    read_images_from_folder(folder_path): Reads images from the specified folder.
    extract_features(images): Extracts features from the given images using a pre-trained ResNet50 model.
    store_features_in_db(images, features): Stores the extracted features in a MongoDB database.
Attributes:
    folder_path (str): The path to the folder containing images.
    images (list): A list of images read from the folder.
    features (list): A list of features extracted from the images.
"""

from services.image_feature_extractor import ImageProcessor as image_processor

folder_path = 'src/data/images'
images = image_processor.read_images_from_folder(folder_path)
features = image_processor.extract_features(images)
image_processor.store_features_in_db(images, features)
