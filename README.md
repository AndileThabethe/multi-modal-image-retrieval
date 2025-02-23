# Multi Modal Retrieval System

This project implements a multi-modal image retrieval system that takes a user query (in text
describing an image) and returns the top-K matching images from a provided dataset. The
system includes a user-friendly front-end where users can input their queries and see the
results.

### System Architecture Diagram
![FinalSystemArchitectureDiagram drawio](https://github.com/user-attachments/assets/0eedef67-0333-4809-95db-f133edfb82e2)

This architecture diagram illustrates a system for retrieving images based on a user's text query. Let's break down the components and the flow of information:

Components:

1. **User Interface**: This is where the user interacts with the system. It is a web page 
   that allows text input and displays results.
2. **User Text Query**: The text query entered by the user, describing the image they are looking for.
2. **Text Feature Extraction** (text_feature_extractor.py): This component processes the user's text query and converts it into a 
   set of numerical features (a vector). The output is "Text features".
3. **Search API** (run.py): This is the core component that receives the text features and uses them to search 
   the image database. It implements a similarity search algorithm.
4. **Similarity Search** (similarity_search.py): This component compares the "Text features" from the query with "Image features" 
    stored in the database to find the most similar images.
5. **Image Features**: Numerical representations of the images, pre-calculated and stored in the database. 
6. **Image Feature Extraction** (image_feature_extractor.py): This component is responsible for processing the raw images (in offline.py as indicated) 
    and generating the "Image features" that are stored in the database.
7. **MongoDB**: The database used to store the "Image features" and potentially other metadata about the images.
8. **Images**: The actual image files. The diagram shows them as being accessed by the "Image Feature Extraction" component.
9. **Retrieved Images**: The images that the system determines to be the most relevant to the user's query. These 
    are returned to the user interface.
10. **Kubernetes**: The system is deployed and orchestrated using Kubernetes, a container orchestration platform. 
11. **Docker/Container**: The containerized nature of the components.

Data Flow:

1. **User Interaction**: The user interacts with the "User Interface" and enters a "User text query".
2. **Text Processing**: The "Text Feature Extraction" component converts the query into "Text features".
3. **Search API Request**: The "Text features" are sent to the "Search API".
4. **Similarity Search**: The "Similarity Search" component compares the query's "Text features" with the 
    "Image features" stored in "MongoDB".
5. **Image Retrieval**: The "Search API" retrieves the most similar "Images" based on the results of the "Similarity Search".
6. **Results Display**: The "Retrieved Images" are sent back to the "User Interface" and displayed to the user.
7. **Offline Processing:** The "Image Feature Extraction" component, running as a batch process (offline.py), 
    extracts "Image features" from the raw "Images" and stores them in "MongoDB".

Production Readiness:

1. **Offline Image Feature Extraction**: The diagram clearly separates the online query processing path from the offline image feature extraction. This is to ensure fast query response times.
2. **Vector Database (Implicit)**: The MongoDB is being used in a way that facilitates efficient similarity search, with possiblity of some form of vector indexing.
3. **Scalability**: The use of Kubernetes strongly suggests the system is designed to be scalable and handle a large number of queries and images.
4. **Containerization**: The Docker icons indicate that each component is packaged in its own container, which improves portability and deployment consistency.

### System Architecture Diagram (catering for people with disabilities)
![SystemArchitectureDiagram drawio](https://github.com/user-attachments/assets/26e69090-1921-47d5-9d28-7e2efa6fe919)

Enhancements to cater for people with disabilities:
- Voice search capability
- Keyboard navigation
- Image input to query
- Customisable user interface e.g Dyslexie, Opendyslexic, Gill Dyslexic, Read Regular, Lexia Readable, or Sylexiad font option for people with Dyslexia
- Make application screen reader compatible
- Alternative text and descriptions that convey the essential information and context of the image
- Visual descriptions and audio narration

### Setup and Installation Instructions
- install MongoDB, mongosh(optional)
- install IDE that supports Python (e.g Visual Studio Code)
- Python 3.12
- requirements.txt file to install required libraries/packages (_pip install -r requirements.txt_ install in terminal)

### How to Run The System
- Run the offline.py file to process the images and store them in the MongoDB
- Run the run.py file to run the application

### How to Run Tests
- To run tests, in terminal type: python -m unittest discover -s tests -p '*_test.py' 

### Assumptions Made
- The data is complete and are images.
