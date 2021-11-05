from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'msg': 'Welcome!'})


@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'msg': 'predict'})
