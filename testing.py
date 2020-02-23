import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time


#Define Path
model_path = 'C:/Users/Hitesh/Desktop/MOdel/All/HexaHive.h5'
model_weights_path = 'C:/Users/Hitesh/Desktop/MOdel/All/HexaHiveweight.h5'
test_path = 'C:/Users/Hitesh/Desktop/HexaHive Test/Test'
#test_path = 'C:/Users/Hitesh/Desktop/13 jan 8.35pm/Test'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 120,120

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)

  a=model.predict_proba(x)
  b = [max(p) for p in a] 

  compare = b[0]

  c = [min(p) for p in a]

  compare1 = c[0]
  
  
  array = model.predict(x,verbose=1)
  print("array",array)
  
  result = array[0]
  print("result",result)
  
  answer = np.argmax(result)
  print('Compare',compare)
  print('Compare1',compare1)
  
  # print('answer',answer)
  if(compare!=1 or compare1!=0):
    answer=4
  if(answer == 0):
    print('Picture is of Animal Waste')
  elif(answer== 1):
    print('Picture is of Bio Hazard')
  elif(answer == 2):
    print('Picture is of Food Waste')
  elif(answer==4):
    print('Not a waste')
  # elif(answer == 3):
  #   print('Picture is of Plane')
  # elif(answer == 4):
  #   print('Picture is of Truck')
  return answer


#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")
