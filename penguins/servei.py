import pickle
from flask import Flask, jsonify, request
import numpy as np

classes = ['Chinstrap', 'Adelie', 'Gentoo']

def predict_single(penguin, dv, sc, model):
    penguin_dv = dv.transform({'island': penguin['island'], 'sex': penguin['sex']})
    penguin_std = sc.transform([[penguin['bill_length_mm'], penguin['bill_depth_mm'], penguin['flipper_length_mm'], penguin['body_mass_g']]])

    penguin_dv_std = np.hstack([penguin_dv, penguin_std])

    y_pred = model.predict(penguin_dv_std)[0]
    y_prob = model.predict_proba(penguin_dv_std)[0][y_pred]

    return (y_pred, y_prob)

def predict(dv, sc, model):
    penguin = request.get_json()
    especie, probabilitat = predict_single(penguin, dv, sc, model)
   
    result = {
        'penguin': classes[especie],
        'probabilitat': float(probabilitat)
    }
    return jsonify(result)

app = Flask('penguins')

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)

if __name__ == '__main__':
    app.run(debug=True, port=8000)