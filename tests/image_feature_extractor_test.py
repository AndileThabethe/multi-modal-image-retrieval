import unittest
from unittest.mock import patch, MagicMock
from src.services.image_feature_extractor import ImageProcessor
from PIL import Image
import numpy as np

class TestImageProcessor(unittest.TestCase):

    @patch('src.services.image_feature_extractor.os.listdir')
    @patch('src.services.image_feature_extractor.Image.open')
    def test_read_images_from_folder(self, mock_open, mock_listdir):

        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image

        images = ImageProcessor.read_images_from_folder('tests/test_images')

        self.assertEqual(len(images), 2)
        mock_open.assert_any_call('tests/test_folder/image1.jpg')
        mock_open.assert_any_call('tests/test_folder/image2.png')

    @patch('src.services.image_feature_extractor.ResNet50')
    @patch('src.services.image_feature_extractor.image.img_to_array')
    @patch('src.services.image_feature_extractor.preprocess_input')
    @patch('src.services.image_feature_extractor.decode_predictions')
    def test_extract_features(self, mock_decode, mock_preprocess, mock_img_to_array, mock_resnet):

        mock_model = MagicMock()
        mock_resnet.return_value = mock_model
        mock_img_to_array.return_value = np.zeros((224, 224, 3))
        mock_preprocess.return_value = np.zeros((1, 224, 224, 3))
        mock_model.predict.return_value = np.zeros((1, 1000))
        mock_decode.return_value = [[('n02124075', 'Egyptian_cat', 0.1)]]

        mock_image = MagicMock(spec=Image.Image)
        mock_image.resize.return_value = mock_image

        features = ImageProcessor.extract_features([mock_image])

        self.assertEqual(len(features), 1)
        self.assertIn('Egyptian_cat', features[0])

    @patch('src.services.image_feature_extractor.MongoClient')
    @patch('src.services.image_feature_extractor.SentenceTransformer')
    def test_store_features_in_db(self, mock_sentence_transformer, mock_mongo_client):

        mock_client = MagicMock()
        mock_mongo_client.return_value = mock_client
        mock_db = mock_client['image_features_db']
        mock_collection = mock_db['features']
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        mock_model.encode.return_value = np.zeros((768,))

        mock_image = MagicMock(spec=Image.Image)
        mock_image.save = MagicMock()

        ImageProcessor.store_features_in_db([mock_image], ['feature'])

        mock_collection.insert_one.assert_called_once()
        args, kwargs = mock_collection.insert_one.call_args
        self.assertIn('image', args[0])
        self.assertIn('feature', args[0])
        self.assertIn('embeddings', args[0])

if __name__ == '__main__':
    unittest.main()