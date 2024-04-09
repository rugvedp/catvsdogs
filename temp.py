import cv2
import keras
from keras.models import load_model

model = load_model('catdog.h5')

image1 = cv2.imread('dog.jpg')
test_img = cv2.resize(image1,(256,256))
process = test_img.reshape((1,256,256,3))

res = model.predict(process)
#print(res[0][0])