from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Sample regression model (replace this with your trained model)
model = pickle.load(open("model.pkl", "rb"))
#preprocessor = pickle.load(open("preprocessing.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    #print(input_data)
    input_data['Children'] = int(input_data['Children'])
    input_data['Additional Features Number'] = int(input_data['Additional Features Number'])

    for column in ['Store Sales', 'Net Weight', 'Package Weight', 'Store Area', 'Frozen Area']:
        input_data[column] = float(input_data[column])

    
    #print(np.array(list(input_data.values()))).reshape(1, -1)
    input_data = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)