import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pickle
import cv2

app = Flask(__name__)

model = pickle.load(open('chest.pk1','rb'))
model._make_predict_function()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():  
    f = request.files["file"]
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    
    x=cv2.imread(file_path)
    x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    x=cv2.resize(x,(480,480))
    x=np.array([cv2.addWeighted (x,4, cv2.GaussianBlur( x, (0,0),277/10) ,-4 ,128)])
    x=np.reshape(x, (1,480,480,1))
    prediction = model.predict(x)
    prediction=[prediction]
    if prediction[0][0][0]==0.0:
        a='Pneumonia Detected'
    else:
        a='No signs of Pneumonia!'


    return render_template('index.html', prediction_text=a)
        
if __name__ == "__main__":
    app.run(debug=True)