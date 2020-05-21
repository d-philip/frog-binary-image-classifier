from flask import Flask, request, jsonify
from model import CNNModel

app = Flask(__name__)
model = CNNModel()
model.deserialize('final_model.h5')

@app.route('/predict/', methods=['POST'])
def predict_class():
    if request.method == 'POST':
        try:
            image = request.get_data(as_text=True)
            image_dict = loads(image)
            img_pix = model.load_image(image_dict['image'], 1)
            pred_dict = model.predict(img_pix)
            return(pred_dict)
        except:
            error = {'error': 'Error with server.'}
            return(error)
    else:
        error = {'error': 'Request method is invalid.'}
        return(error)
