from flask import Flask, render_template, request, jsonify
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
print("app")

# Load the model
model = joblib.load('kelly-model.pkl')

# Load the language models from the local directory
tokenizer = AutoTokenizer.from_pretrained("./local_models/argument-classifier")
arg_model = AutoModelForSequenceClassification.from_pretrained("./local_models/argument-classifier")
sim_model = SentenceTransformer('./local_models/sentence-similarity')


# Function to calculate similarity between argument and topic
def calculate_similarity(argument, topic):
    # Encode the sentences
    topic_embedding = sim_model.encode(topic, convert_to_tensor=True)
    argument_embedding = sim_model.encode(argument, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_score = util.pytorch_cos_sim(topic_embedding, argument_embedding)
    
    # Return the similarity score
    return cosine_score.item()


# Function to classify an argument and return probabilities
def classify_argument(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Get the model output
    outputs = arg_model(**inputs)

    # Get the predicted probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Return probabilities as a list
    return probs.tolist()[0][1]


# Function to predict the score of an argument
def predict_argument_score(argument, topic):
    argument_prob = classify_argument(argument)
    similarity_score = calculate_similarity(argument, topic)

    # Prepare the input features for the model
    X_test = [[argument_prob, similarity_score]]
    
    # Predict the accuracy score
    accuracy_score = model.predict(X_test)
    
    return accuracy_score[0]  # Return the predicted accuracy score


# Define route for the home page with the form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the argument and topic from the form
        argument = request.form['argument']
        topic = request.form['topic']
        
        # Use the model to make predictions and get the argument quality score
        accuracy_score = predict_argument_score(argument, topic)
        
        # Return the result as JSON
        return jsonify({'score': accuracy_score})
    else:
        # Render the HTML form
        return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
