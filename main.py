from flask import Flask, render_template, request
import pandas as pd
import pickle
from flask_paginate import Pagination, get_page_args

app = Flask(__name__)   

# Load the pre-trained model
model = pickle.load(open('predict_manhattan.pkl', 'rb'))
modelEU = pickle.load(open('predict_euclidean.pkl', 'rb'))
modelcr = pickle.load(open('predict_canberra.pkl', 'rb'))
modelcs = pickle.load(open('predict_chebyshev.pkl', 'rb'))
modelms = pickle.load(open('predict_minkowski.pkl', 'rb'))

# Load the dataset once to avoid multiple I/O operations
data = pd.read_csv('German.data2.csv')

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

def get_data(offset=0, per_page=10):
    return data.iloc[offset: offset + per_page].to_dict(orient='records')


# ... existing code ...

@app.route("/dataset")
def dataset():
    page, per_page, offset = get_page_args(page_parameter='page', per_page_parameter='per_page')
    total = data.shape[0]
    items = get_data(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total, css_framework='bootstrap4')

    return render_template('dataset.html', items=items, page=page, per_page=per_page, pagination=pagination)



@app.route('/klasifikasi')
def klasifikasi():
    return render_template('klasifikasi.html')



@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # List of form fields
            form_fields = [
                'Checking Account', 'Duration', 'Credit History', 'Purpose', 'Credit Amount',
                'Savings Account', 'Employment Since', 'Installment Rate', 'Personal Status',
                'Other Debtors', 'Residence', 'Property', 'Age', 'Other Installment Plans',
                'Housing', 'Existing Credit', 'Job', 'Maintenance', 'Telephone', 'Foreign Worker'
            ]

            # Extract form data using a loop
            form_data = [int(request.form[field]) for field in form_fields]

            # Make predictions
            predict = model.predict([form_data])
            output = predict[0]

            predictEU = modelEU.predict([form_data])
            outputEU = predictEU[0]

            predictCR = modelcr.predict([form_data])
            outputCR = predictCR[0]

            predictCS = modelcs.predict([form_data])
            outputCS = predictCS[0]

            predictMS = modelms.predict([form_data])
            outputMS = predictMS[0]



            # Convert numerical output to categorical label
            label = "Good Credit" if output == 0 else "Bad Credit"
            formatted = f'Dengan data yang telah anda input, maka masuk kedalam kelas : {output}'
            return render_template('klasifikasi.html', euclidean = outputEU, canberra = outputCR, chebyshev = outputCS, manhattan = output, minkowski=outputMS ,prediction="{}".format(formatted), prediction1="{}".format(label))

        except ValueError:
            return render_template('klasifikasi.html', error='Mohon maaf, data yang anda input harus berupa angka')

    return render_template('klasifikasi.html')

@app.route('/evaluasi')
def evaluasi():
    return render_template('evaluasi.html')

if __name__ == "__main__":
    app.run(debug=True)
