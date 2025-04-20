from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("powerconsumption.csv")
df = df.dropna()
df['Datetime'] = pd.to_datetime(df['Datetime'])

from scipy import stats
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[float, int]))) < 3).all(axis=1)]

df['Hour'] = df['Datetime'].dt.hour
df['Day'] = df['Datetime'].dt.day
df['Month'] = df['Datetime'].dt.month
df['Year'] = df['Datetime'].dt.year

# Features and target
X = df[['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']]
y = df['PowerConsumption_Zone1']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model for future use
joblib.dump(model, 'power_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    temp = float(data['temperature'])
    humidity = float(data['humidity'])
    wind = float(data['wind'])
    gdf = float(data['general_diffuse'])
    df_ = float(data['diffuse'])

    model = joblib.load('power_model.pkl')
    prediction = model.predict([[temp, humidity, wind, gdf, df_]])[0]

    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
