from flask import Flask, request
from model import CNNModel

app = Flask(__name__)
model = CNNModel()
model.deserialize('final_model.h5')

@app.route('/predict/', methods=['POST'])
def predict_class():
    if request.method== 'POST':
        print (request.files['file'])
        if request.files:
            image = request.files['file']
            pred_dict = model.predict(image)
            return pred_dict
        abort(404)
