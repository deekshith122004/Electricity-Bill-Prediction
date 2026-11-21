
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Read CSV file
data = pd.read_csv('electricity_bill_dataset.csv')

# Input and output
X = data[['Units_Consumed', 'Number_of_Appliances',
          'Family_Members', 'AC_Hours']]

Y = data['Bill_Amount']

# Train model
model = LinearRegression()
model.fit(X, Y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    units = float(request.form['units'])
    appliances = int(request.form['appliances'])
    family = int(request.form['family'])
    ac = float(request.form['ac'])

    new_data = pd.DataFrame({
        'Units_Consumed': [units],
        'Number_of_Appliances': [appliances],
        'Family_Members': [family],
        'AC_Hours': [ac]
    })

    result = model.predict(new_data)

    bill = round(result[0], 2)

    return render_template('index.html', bill=bill)


if __name__ == '__main__':
    app.run(debug=True)

