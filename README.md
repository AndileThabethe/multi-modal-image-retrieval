# Multi Modal Retrieval System

This project implements a multi-modal image retrieval system that takes a user query (in text
describing an image) and returns the top-K matching images from a provided dataset. The
system includes a user-friendly front-end where users can input their queries and see the
results.

### System Architecture Diagram
![FinalSystemArchitectureDiagram drawio](https://github.com/user-attachments/assets/0eedef67-0333-4809-95db-f133edfb82e2)
1. The images are extracted from source and processed to obtain the image features.
2. The extracted features and images are stored in a mongoDB.
3. The user inputs a description to query.
4. The user text query goes to the search API that routes the query to the text teature extraction process to extract the text features.
5. The search similarity reads the mongoDB and does a similarity search to find the K images that match the description.
6. The K images are sent to the search API and displayed on the user interface.

### System Architecture Diagram (catering for people with disabilities)
![SystemArchitectureDiagram drawio](https://github.com/user-attachments/assets/26e69090-1921-47d5-9d28-7e2efa6fe919)

Enhancements to cater for people with disabilities:
- Speech-to-Text input to query
- Image input to query
- Dyslexie, Opendyslexic, Gill Dyslexic, Read Regular, Lexia Readable, or Sylexiad font option 

### Setup and Installation Instructions
- install MongoDB, mongosh(optional)
- install IDE that supports Python

### How to Run The System
- Run the offline.py file to process the images and store them in the MongoDB
- Run the run.py file to run the application

### How to Run Tests

### Assumptions Made
