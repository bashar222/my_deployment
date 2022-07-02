port = 8008
from flask import Flask
from flask import request, jsonify
import pickle
import numpy as np
import sklearn
import json

m = pickle.load(open('covid_19_detection_RF', 'rb'))
app = Flask(__name__)


@app.route('/predict', methods=['POST','GET'])
def predict():
    data = json.loads(json.loads(request.data))
    print(data)
    print(type(data))
    sex = data['sex']
    intubed = data['intubed']
    pneumonia = data['pneumonia']
    age = data['age']
    pregnancy = data['pregnancy']
    diabetes = data['diabetes']
    copd = data['copd']
    asthma = data['asthma']
    inmsupr = data['inmsupr']
    hypertension = data['hypertension']
    other_disease = data['other_disease']
    cardiovascular = data['cardiovascular']
    obesity = data['obesity']
    renal_chronic = data['renal_chronic']
    tobacco = data['tobacco']
    icu = data['icu']





    input_query = np.asarray(
        [[sex, intubed, pneumonia, age, pregnancy,
         diabetes, copd, asthma, inmsupr, hypertension,
         other_disease, cardiovascular, obesity, renal_chronic,
         tobacco, icu]])


    print(input_query)
    result = m.predict(input_query)[0]
    print(result)
    print(jsonify({'patient state':str(result)}))
    return jsonify({'patient state':str(result)})

@app.route('/', methods=['POST','GET'])
def test():
    resp = jsonify(success=True)
    return resp

if __name__ == '__main__':
    app.run(debug=True, port=port)
