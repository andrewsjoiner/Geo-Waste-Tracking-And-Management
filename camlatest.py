import urllib
import urllib.request
import cv2
import numpy as np
import cv2 
import os 
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

#model_path = 'fine_work.mopdel' ##clean+waste

# model_path = 'af.model' ##animal+food

#model_path = 'dp.model' ##drainage+plastic

# model_path = 'mix.model' ##mix model

#Define Path
model_path = 'C:/Users/Hitesh/Desktop/MOdel/All/HexaHive.h5'
model_weights_path = 'C:/Users/Hitesh/Desktop/MOdel/All/HexaHiveweight.h5'
test_path = 'C:/Users/Hitesh/Desktop/HexaHive Test/Test'
#test_path = 'C:/Users/Hitesh/Desktop/13 jan 8.35pm/Test'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

url = "http://192.168.43.1:8080/shot.jpg"
cam = cv2.VideoCapture('http://192.168.43.1:8080/shot.jpg')
imgResp = urllib.request.urlopen(url)

while True:
	ret,frame = cam.read() 
	imgResp = urllib.request.urlopen(url)
	imgnp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	img = cv2.imdecode(imgnp,-1)
	org = img
	org = cv2.resize(org,(640,640))
	img = cv2.resize(img,(120,120))
	#print(img)
	# org = img
	# org = cv2.resize(org,(640,640))
	img = np.expand_dims(img, axis=0)

	a=model.predict_proba(img)
	b = [max(p) for p in a]
	compare = b[0]
	a=model.predict_proba(img)
	b = [max(p) for p in a] 

	compare = b[0]

	c = [min(p) for p in a]

	compare1 = c[0]
	# result = model.predict(img)
	answer = model.predict(img)
	# if result[0][0] == 0:
	# 	prediction = 'Clean: Location: 20.593683L,78.962883L'
	# else:
	# 	prediction='Waste: Location: 20.593683L,78.962883L'

	# if result[0][0] == 0:
	# 	prediction = 'Special Waste: Animal dead Location: 20.593683L,78.962883L'
	# else:
	# 	prediction='Food Waste: Location: 20.593683L,78.962883L'

	answer = np.argmax(answer)
	if(compare!=1 or compare1!=0):
		answer=4
	if(answer == 0):
		print('Picture is of Animal Dead')
	elif(answer== 1):
		print('Picture is of Bio Haradous')
	elif(answer == 2):
		print('Picture is of Food Waste')
	elif(answer):
		print('Not a Waste')

	

	# if result[0][0] == 0:
	# 	prediction = 'Drainage: Location: 20.593683L,78.962883L'
	# else:
	# 	prediction='Plastic: Location: 20.593683L,78.962883L'

	#print(prediction)

	# img = np.expand_dims(img, axis=0)

	#img = img[...,np.newaxis]

	# org = img
	# img = cv2.resize(img,(640,640))
	# img = image.img_to_array(img)
	# img = np.expand_dims(img, axis=0)

	#print(type(img))
	cv2.imshow('test', org)
	if cv2.waitKey(1) == 27: ##Check whether user has pressed esc key or not
		break