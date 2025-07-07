import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model('model.h5')

def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28,28))
    img = ImageOps.invert(img)
    img = np.array(img).astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')
    file = request.files['file']
    if file.filename == '':
        return redirect('/')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        img = prepare_image(filepath)
        pred = model.predict(img)
        digit = np.argmax(pred)
        return render_template('index.html', prediction=digit, image_url=url_for('static', filename='uploads/' + file.filename))
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

