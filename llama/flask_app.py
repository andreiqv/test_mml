from flask import Flask
from flask import request
from inference_llama2 import inference

# Create a Flask application instance
app = Flask(__name__)

# Define a route for the home page
@app.route('/', methods=['GET', 'POST'])
def inference_model():
    question = request.form.get('question')
    result = inference(question)
    return result

# Run the Flask application
if __name__ == '__main__':
    app.run()