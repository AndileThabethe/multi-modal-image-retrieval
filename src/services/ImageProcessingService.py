import os
from PIL import Image

def read_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)
    return images

# Example usage
folder_path = '/src/data/images'
images = read_images_from_folder(folder_path)
for img in images:
    img.show()  # This will open each image