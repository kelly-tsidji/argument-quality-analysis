# Argument Quality Prediction Project

## Overview

This project aims to predict the quality of an argument based on its content and the given topic. The core components of this project include a dataset, machine learning models, and a web interface for user interaction.

### Dataset

The dataset used for this project is the **IBM Debater(R) - IBM-ArgQ-Rank-30kArgs**. It contains 30,497 arguments labeled for quality and stance, divided into train, dev, and test sets.

### Model and Approach

1. **Argument Classification**: A pre-trained argument classification model is used to determine the likelihood that a given text is a valid argument.
2. **Similarity Calculation**: Sentence embeddings are used to calculate the cosine similarity between the argument and the topic.
3. **Quality Prediction**: A Gradient Boosting Regressor is trained on the features (argument probability and similarity score) to predict the quality of the argument.

### Web Interface

A Flask web application is provided to allow users to input an argument and a topic, and receive a predicted quality score for the argument.

## Files and Directory Structure

- `dataset_info.txt`: Information about the dataset used.
- `model.ipynb`: Jupyter notebook containing the data processing, model training, and evaluation.
- `argument-accuracy.py`: Flask application for serving the model and providing a web interface.
- `form.html`: HTML file for the web interface.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

### Step-by-Step Setup

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Save Pre-trained Models**

   Run: `python download-models.py` to download and save the pre-trained models locally in the `local_models` directory.

5. **Run the Flask Application**

   ```bash
   python argument-accuracy.py
   ```

   The application will be available at `http://127.0.0.1:5000`.

## Usage

1. **Start the Flask Application**

   Ensure the Flask server is running (`python argument-accuracy.py`).

2. **Open the Web Interface**

   Open a web browser and navigate to `http://127.0.0.1:5000`.

3. **Enter Argument and Topic**

   Fill in the form with an argument and a topic, then submit the form.

4. **View the Result**

   The application will display the predicted quality score and category of the argument.

## Project Components

### dataset_info.txt

Contains details about the dataset including its name, version, release date, and the structure of the data.

### model.ipynb

Jupyter notebook for:

- Loading the dataset
- Classifying arguments and calculating similarity scores
- Training the Gradient Boosting Regressor
- Evaluating the model performance
- Saving the trained models

### argument-accuracy.py

Flask application script to:

- Load the trained models
- Define routes for the web interface
- Handle form submissions
- Predict and return argument quality scores

### form.html

HTML file providing the user interface for inputting arguments and topics, and displaying the predicted quality score.

## License

The dataset and some components of this project are licensed under the Creative Commons Attribution-ShareAlike license (CC-BY-SA). Please refer to the dataset_info.txt for more details.

## Acknowledgements

- IBM Debater(R) team for providing the dataset.
- Authors of the paper "A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis" (AAAI 2020).

## References

If you use this project or dataset in your research, please cite:

```
A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis
Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit Aharonov and Noam Slonim
AAAI 2020
```

For further details, please refer to the individual files and their comments.
