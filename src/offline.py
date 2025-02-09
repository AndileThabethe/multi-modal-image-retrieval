# This file is utilized to extract features from images and store them in the database. 
# The features are extracted using a pre-trained ResNet50 model. The images are read from a folder, 
# and the features are stored in a MongoDB database. The code is executed by running the app.py file. 
# The features can then be used for image similarity search or other image processing tasks.

from services.imageprocessor import ImageProcessor as image_processor

folder_path = 'src/data/images'
images = image_processor.read_images_from_folder(folder_path)
features = image_processor.extract_features(images)
image_processor.store_features_in_db(images, features)
