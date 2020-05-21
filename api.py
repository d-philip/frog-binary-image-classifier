from flask import Flask, request, jsonify
from model import CNNModel
from json import loads

app = Flask(__name__)
model = CNNModel()
model.deserialize('final_model.h5')

@app.route('/predict/', methods=['POST'])
def predict_class():
    try:
        image = request.get_data(as_text=True)
        image_dict = loads(image)
        img_pix = model.load_image(image_dict['image'], 1)
        pred_dict = model.predict(img_pix)
        return(pred_dict)
    except Exception as err:
        error = {'error': str(err)}
        return(error)
