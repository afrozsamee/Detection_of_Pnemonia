import pickle
import numpy as np
import cv2


model = pickle.load(open('chest.pk1','rb'))

img_path='C:\\Users\\Mariah\\Detection of Pneumonia\\uploads\\10089068.jpeg'
x=cv2.imread(img_path)
x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
x=cv2.resize(x,(480,480))
x=np.array([cv2.addWeighted (x,4, cv2.GaussianBlur( x, (0,0),277/10) ,-4 ,128)])
x=np.reshape(x, (1,480,480,1))
prediction = model.predict(x)
prediction=[prediction]
if prediction[0][0][0]==0.0:
    print('Pneumonia Detected')
else:
    print('No signs of Pneumonia!')

