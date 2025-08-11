from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load preprocessing tools and models
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("protocol_encoder.pkl")
catboost_model = joblib.load("catboost_model.pkl")

models = {
    "catboost": catboost_model
}

def preprocess_csv(csv_path, scaler, label_encoder):
    df = pd.read_csv(csv_path)

    # Handle Protocol
    if 'Protocol' in df.columns:
        known_classes = list(label_encoder.classes_)
        df['Protocol'] = df['Protocol'].apply(lambda x: x if x in known_classes else known_classes[0])
        df['Protocol_Encoded'] = label_encoder.transform(df['Protocol'])
    else:
        df['Protocol_Encoded'] = 0

    for col in ['Time', 'Length']:
        if col not in df.columns:
            df[col] = 0

    df[['Time_Normalized', 'Length_Normalized']] = scaler.transform(df[['Time', 'Length']])

    # Final features for prediction
    X_processed = df[['Protocol_Encoded', 'Time_Normalized', 'Length_Normalized']]

    return X_processed, df[['Protocol', 'Time', 'Length']]  # return original for display

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    columns = []
    detailed_data = []
    filename = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                X_processed, original_data = preprocess_csv(filepath, scaler, label_encoder)

                for name, model in models.items():
                    predictions = model.predict(X_processed)
                    intrusion_count = int(sum(predictions))
                    results[name] = f"{intrusion_count} intrusion(s) detected out of {len(predictions)} packets."

                    # Append predictions to original features
                    original_data['Prediction'] = predictions
                    columns = original_data.columns.tolist()
                    detailed_data = original_data.to_dict(orient='records')

            except Exception as e:
                results = {"Error": str(e)}

    return render_template('index.html', results=results, filename=filename, columns=columns,detailed_data=detailed_data)

if __name__ == '__main__':
    app.run(debug=True)