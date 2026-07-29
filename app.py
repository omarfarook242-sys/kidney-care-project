from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# সেভ হওয়া মডেল লোড করা
model = joblib.load('ckd_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # ইউজারের ইনপুট গ্রহণ
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        sc = float(request.form['sc'])
        hemo = float(request.form['hemo'])

        input_data = np.array([[age, bp, sg, al, sc, hemo]])
        
        # মডেল দিয়ে প্রেডিকশন
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "সতর্কতা: ক্রনিক কিডনি ডিজিজ (CKD) এর লক্ষণ পাওয়া গেছে।"
        else:
            result = "অভিনন্দন! কিডনি ডিজিজের কোনো লক্ষণ পাওয়া যায়নি।"

        return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)