from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# সেভ হওয়া মডেল লোড করার নিরাপদ ব্যবস্থা
model = None
model_path = 'ckd_model.pkl'

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f" মডেল লোড করতে সমস্যা হয়েছে: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # মডেল না থাকলে যাতে ক্র্যাশ না করে
        if model is None:
            return render_template('index.html', prediction_text="ত্রুটি: সার্ভারে মডেল ফাইল (ckd_model.pkl) পাওয়া যায়নি!")

        # ইউজারের ইনপুট গ্রহণ
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        sc = float(request.form['sc'])
        hemo = float(request.form['hemo'])

        input_data = np.array([[age, bp, sg, al, sc, hemo]])
        
        # মডেল দিয়ে প্রেডিকশন
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "সতর্কতা: ক্রনিক কিডনি ডিজিজ (CKD) এর লক্ষণ পাওয়া গেছে। দ্রুত চিকিৎসকের পরামর্শ নিন।"
        else:
            result = "অভিনন্দন! কিডনি ডিজিজের কোনো লক্ষণ পাওয়া যায়নি। আপনার স্বাস্থ্য স্বাভাবিক রয়েছে।"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"ইনপুট ডেটায় ত্রুটি রয়েছে: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
