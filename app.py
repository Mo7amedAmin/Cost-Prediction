from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request

    if input_data.is_json:
        input_data = input_data.get_json(force=True)
    else:
        input_data = input_data.form.to_dict()

    input_data = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    return render_template('index.html', prediction_text=prediction[0])
    
if __name__ == '__main__':
    app.run(debug=True)