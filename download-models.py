from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Download and save argument classification model
arg_model_name = "addy88/argument-classifier"
arg_tokenizer = AutoTokenizer.from_pretrained(arg_model_name)
arg_model = AutoModelForSequenceClassification.from_pretrained(arg_model_name)
arg_tokenizer.save_pretrained("./local_models/argument-classifier")
arg_model.save_pretrained("./local_models/argument-classifier")

# Download and save sentence similarity model
sim_model_name = "annakotarba/sentence-similarity"
sim_model = SentenceTransformer(sim_model_name)
sim_model.save("./local_models/sentence-similarity")