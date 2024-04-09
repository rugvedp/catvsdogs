from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

model = load_model('catdog.h5')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image_file' not in request.files:
            return 'No file part'
        file = request.files['image_file']
        npimg = np.fromstring(file.read(), np.uint8)
        image1 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        resized_img = cv2.resize(image1, (256, 256))
        processed_img = resized_img.reshape((1, 256, 256, 3))

        res = model.predict(processed_img)
        print(res[0][0])
        final =''
        if res[0][0] == 1.0:
            final = 'Dog'
        else:
            final = 'Cat'
        return render_template('index.html', prediction_text= 'Your image is of  {}' . format(final) ) 
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
