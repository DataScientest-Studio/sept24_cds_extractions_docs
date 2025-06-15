DOCUMENT CLASSIFICATION
==============================

The project is set in the context of increasing digitization of documents and the development of artificial intelligence. Companies, especially those in the insurance sector, are seeking to automate the classification of their documents (such as birth certificates, sales contracts, etc.) to improve efficiency and accuracy.

Technologies such as OCR (Optical Character Recognition), NLP (Natural Language Processing), CV (Computer Vision), or a hybrid approach combining NLP and CV offer promising solutions to address this challenge.


Project Organization
------------

    ├── README.md
    │
    ├── models                          <- Contains trained CNN model and model training summaries
    │   └── training_history            <- Contains CNN model training summaries
    │
    ├── notebooks                       <- Jupyter notebooks
    │   ├── CNN_analysis.ipynb          <- Contains the analysis of the results for different CNN models     
    │   ├── data_pixels_stats.ipynb     <- Contains the data exploration and visualitations for the Computer vision approach
    │   └── text_based_classifier_analysis.ipynb <- Contains the data exploration and model training for the text minning approach
    │
    ├── reports            <- Contains the final report of this project
    │
    ├── requirements.txt   
    │
    ├── src                <- Source code for the streamlit app
    │   └── streamlit      
    │       └── app.py
    ├── tools              <- Reusable fonctions for data exploration and model training
    │   ├── callback.py    <- Define customs tensorflow callbacks
    │   └── tools.py       <- Define useful fonction for data exploration and preprocessing
    │
    ├── training_scripts        <- Training scripts
    │   ├── predict_clip.py     <- Script to train and predict using CLIP
    │   └── train_CNN_models.py <- Script to train all tested CNN architecture
--------
