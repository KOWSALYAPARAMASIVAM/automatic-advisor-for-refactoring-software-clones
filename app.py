from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    clone_type = request.form['Clone_Type']
    code_complexity = int(request.form['Code_Complexity'])
    
    clone_type_map = {'Type1': 0, 'Type2': 1, 'Type3': 2}
    clone_type_code = clone_type_map[clone_type]

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict([[clone_type_code, code_complexity]])[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
