from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import math
app = Flask(__name__)

df=pd.read_csv("D:/Data_set/Mumbai House Prices.csv.zip")
model = pickle.load(open("D:/Model/house_price.pkl", "rb"))


type_list = sorted(df['type'].unique())
type_map = {v: i for i, v in enumerate(type_list)}


regions = sorted(df['region'].unique())
region_map = {region: i for i, region in enumerate(regions)}


price_unit_list = sorted(df['price_unit'].unique())
price_unit_map = {v: i for i, v in enumerate(price_unit_list)}


status_list = sorted(df['status'].unique())
status_map = {v: i for i, v in enumerate(status_list)}


age_list = sorted(df['age'].unique())
age_map = {v: i for i, v in enumerate(age_list)}
# Home page
@app.route("/")
def index():
    return render_template('house.html',regions=regions)

@app.route("/predict", methods=['POST'])
def predict_route():
    bhk = float(request.form['bhk'])
    type_ = request.form['type']
    area = float(request.form['area'])
    price_unit = request.form['price_unit']
    region = request.form['region']
    status = request.form['status']
    age = request.form['age']

    features = [[
        float(bhk),
        float(type_map.get(type_.strip(), 0)),
        float(area),
        float(price_unit_map.get(price_unit.strip(), 0)),
        float(region_map.get(region.strip(), 0)),
        float(status_map.get(status.strip(), 0)),
        float(age_map.get(age.strip(), 0))
    ]]
    prediction = model.predict(features)

    final_price = prediction[0]

    return render_template('house.html', prediction_text=f"Price: ₹{round(math.fabs(final_price),2)}")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)