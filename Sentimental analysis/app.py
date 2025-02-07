from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import emoji  # Added for handling emojis

app = Flask(__name__)

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)
 
# Function to preprocess the input text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Convert emojis to text descriptions
    text = emoji.demojize(text)  # Converts emoji to a format like :grinning_face:
    
    # Lowercase, remove punctuation, and remove numbers
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('sentiment_analysis.html') 

# Predict sentiment using the model
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    try:
        # Get review text from the request
        review_text = request.json.get('review')

        # Preprocess the review text
        cleaned_review = preprocess_text(review_text)

        # Transform the review text using the vectorizer
        review_vector = vectorizer.transform([cleaned_review])

        # Predict sentiment using the model
        prediction = model.predict(review_vector)

        # Return the result as a JSON response
        if prediction[0] == 2:
            sentiment = 'positive'
        elif prediction[0] == 1:
            sentiment = 'neutral'
        else:
            sentiment = 'negative'

        return jsonify({'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    
    app.run(debug=True)
