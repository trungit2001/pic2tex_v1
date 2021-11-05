import base64
import pic2tex

from flask import Flask, jsonify, request


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'bin'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return jsonify({'msg': 'Welcome!'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            decode = open('dataset/img.png', 'wb')
            decode.write(base64.b64decode(file.read()))
            decode.close()

            tex = pic2tex.get_predict('dataset/img.png')

            return jsonify({'msg': 'predicted', 'tex': tex})
        except:
            return jsonify({'error': 'error during prediction'})
