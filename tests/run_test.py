import unittest
from flask import Flask
from src.run import app

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page_loads(self):
        response = self.app.get('/Home')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Lookup an image', response.data)

    def test_search_functionality(self):
        response = self.app.post('/Home', data=dict(search='test query'), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Lookup an image', response.data)
        
if __name__ == '__main__':
    unittest.main()