import os
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)

        if 'type' in df.columns:
            df['type'] = df['type'].astype('category')
            df['type_code'] = df['type'].cat.codes
        else:
            return "Missing 'type' column."

        feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'type_code']
        if not all(col in df.columns for col in feature_cols):
            return "Missing required columns."

        X = df[feature_cols]
        predictions = model.predict(X)
        df['isSuspicious'] = predictions
        suspicious_df = df[df['isSuspicious'] == 1]

        return render_template('index.html', suspicious=suspicious_df.to_dict(orient='records'))

    return "Invalid file format."

if __name__ == '__main__':
    app.run(debug=True)
