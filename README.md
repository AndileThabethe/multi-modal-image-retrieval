# Multi Modal Retrieval System

This project implements a multi-modal image retrieval system that takes a user query (in text
describing an image) and returns the top-K matching images from a provided dataset. The
system includes a user-friendly front-end where users can input their queries and see the
results.

### System Architecture Diagram
![FinalSystemArchitectureDiagram drawio](https://github.com/user-attachments/assets/0eedef67-0333-4809-95db-f133edfb82e2)

This architecture diagram illustrates a system for retrieving images based on a user's text query. Let's break down the components and the flow of information:

Components:

    User Interface: This is where the user interacts with the system. It could be a web page, a mobile app, or any other interface that allows text input and displays results.
    User Text Query: The text query entered by the user, describing the image they are looking for.
    Text Feature Extraction: This component processes the user's text query and converts it into a set of numerical features (a vector). This is typically done using techniques like TF-IDF, word embeddings (Word2Vec, GloVe), or sentence embeddings (Sentence-BERT). The output is "Text features".
    Search API: This is the core component that receives the text features and uses them to search the image database. It likely implements a similarity search algorithm.
    Similarity Search: This component compares the "Text features" from the query with "Image features" stored in the database to find the most similar images. Cosine similarity or other distance metrics are commonly used.
    Image Features: Numerical representations of the images, pre-calculated and stored in the database. These are typically extracted using Convolutional Neural Networks (CNNs).
    Image Feature Extraction: This component is responsible for processing the raw images (likely in offline.py as indicated) and generating the "Image features" that are stored in the database.
    MongoDB: The database used to store the "Image features" and potentially other metadata about the images.
    Images: The actual image files. The diagram shows them as being accessed by the "Image Feature Extraction" component.
    Retrieved Images: The images that the system determines to be the most relevant to the user's query. These are returned to the user interface.
    Kubernetes: Indicates that the system is likely deployed and orchestrated using Kubernetes, a container orchestration platform. This suggests a scalable and containerized architecture.
    Docker/Container Icons: The presence of Docker icons reinforces the containerized nature of the components.

Data Flow:

    User Interaction: The user interacts with the "User Interface" and enters a "User text query".
    Text Processing: The "Text Feature Extraction" component converts the query into "Text features".
    Search API Request: The "Text features" are sent to the "Search API".
    Similarity Search: The "Similarity Search" component compares the query's "Text features" with the "Image features" stored in "MongoDB".
    Image Retrieval: The "Search API" retrieves the most similar "Images" based on the results of the "Similarity Search".
    Results Display: The "Retrieved Images" are sent back to the "User Interface" and displayed to the user.
    Offline Processing: The "Image Feature Extraction" component, likely running as a batch process (offline.py), extracts "Image features" from the raw "Images" and stores them in "MongoDB".

Key Observations:

    Offline Feature Extraction: The diagram clearly separates the online query processing path from the offline image feature extraction. This is a common practice to ensure fast query response times.
    Vector Database (Implicit): While not explicitly stated, the architecture suggests that MongoDB is being used in a way that facilitates efficient similarity search, likely with some form of vector indexing.
    Scalability: The use of Kubernetes strongly suggests that the system is designed to be scalable and handle a large number of queries and images.
    Containerization: The Docker icons indicate that each component is likely packaged in its own container, which improves portability and deployment consistency.

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
