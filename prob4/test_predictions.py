import numpy as np
import pickle
import os
import data_loader

print("Sport vs Politics Prediction Tool\n")

# Check if trained model exists
if not os.path.exists('results/best_model.pkl') or not os.path.exists('results/best_vectorizer.pkl'):
    print("Error: Trained model not found!")
    print("Please run classifier.py first to train the models.")
    exit(1)

# Load best model and vectorizer
print("Loading best model...")
with open('results/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('results/best_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load model info
if os.path.exists('results/best_model_info.txt'):
    with open('results/best_model_info.txt', 'r') as f:
        print(f.read())

print("Ready for predictions\n")

# Interactive prediction
while True:
    print("\nEnter a text document (or 'exit' to quit):")
    user_text = input("> ")

    if user_text.lower() == 'exit':
        print("Goodbye!")
        break

    if user_text.strip() == '':
        print("Please enter some text")
        continue

    # Clean and predict
    cleaned_text = data_loader.clean_text(user_text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]

    print(f"Prediction: {prediction}")
