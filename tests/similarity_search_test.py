import unittest
from unittest.mock import patch, MagicMock
from src.services.similarity_search import data_query, image_to_base64, semantic_similarity
from PIL import Image
import numpy as np
import torch

class TestSimilaritySearch(unittest.TestCase):

    @patch('src.services.similarity_search.db')
    @patch('src.services.similarity_search.model.encode')
    @patch('src.services.similarity_search.util.pytorch_cos_sim')
    def test_data_query(self, mock_cos_sim, mock_encode, mock_db):

        mock_db["features"].find.return_value = [
            {"_id": "1", "embeddings": [0.1, 0.2, 0.3]},
            {"_id": "2", "embeddings": [0.4, 0.5, 0.6]}
        ]
        mock_encode.return_value = torch.tensor([0.1, 0.2, 0.3])
        mock_cos_sim.return_value = torch.tensor([[0.9, 0.8]])

        mock_db["features"].find.return_value = [
            {"_id": "1", "image": b"fake_image_data_1"},
            {"_id": "2", "image": b"fake_image_data_2"}
        ]

        with patch('src.services.similarity_search.Image.open') as mock_open:
            mock_image = MagicMock(spec=Image.Image)
            mock_open.return_value = mock_image

            result = data_query("test query")

            self.assertEqual(len(result), 2)
            mock_open.assert_any_call(io.BytesIO(b"fake_image_data_1"))
            mock_open.assert_any_call(io.BytesIO(b"fake_image_data_2"))

    def test_image_to_base64(self):

        mock_image = MagicMock(spec=Image.Image)
        mock_image.save = MagicMock()

        result = image_to_base64(mock_image)

        self.assertIsInstance(result, str)
        mock_image.save.assert_called_once()

    @patch('src.services.similarity_search.model.encode')
    @patch('src.services.similarity_search.util.pytorch_cos_sim')
    def test_semantic_similarity(self, mock_cos_sim, mock_encode):

        mock_encode.side_effect = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])]
        mock_cos_sim.return_value = torch.tensor([[0.9]])

        result = semantic_similarity("text1", [0.4, 0.5, 0.6])

        self.assertEqual(result, 0.9)
        mock_encode.assert_any_call("text1", convert_to_tensor=True)
        mock_cos_sim.assert_called_once()

if __name__ == '__main__':
    unittest.main()