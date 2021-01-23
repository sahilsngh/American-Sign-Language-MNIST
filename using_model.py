import sys
import cv2
import time
import warnings
import numpy as np
import tensorflow as tf 
warnings.filterwarnings("ignore")

HIEGHT = 28
WIDTH = 28
ALPHA = ['a','b','c','d','e','f','g','h','i','k','l',
'm','n','o','p','r','s','t','u','v','w','x','y','z']
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1
color = (255, 0, 0) 
thickness = 2


def prediction(img):

	# Recreate the exact same model, including its weights and the optimizer
	path = f'{os.getcwd()}/'.replace('\\', '/')
	model_path = path + "asl_mnist_95%_acc.h5"
	model = tf.keras.models.load_model(model_path)

	# Show the model architecture
	# model.summary()

	# print(len(img)) # 480
	# print(img.shape) # 480*640

	x = preprocepccessed_img(img)
	# print(f"{x.shape}, {type(x)}")

	pre = model.predict([x.reshape(-1,28,28,1)], batch_size=1)
	return pre

def preprocepccessed_img(img):
	image = img

	# image = tf.image.decode_jpeg(image)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (28,28))

	# image = np.array(image)
	# image = [image.reshape(28,28,1)]
	# image = tf.image.convert_image_dtype(image, tf.float32)
	# image = tf.image.resize(image, [28, 28])
	
	return image

cam = cv2.VideoCapture(0)

for i in list(range(4))[::-1]:
	print(i+1)
	time.sleep(1)

while True:

	ret, frame = cam.read()
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.flip(frame, 1)
	# predict = prediction(gray)
	cv2.imshow('test', gray)


	k = cv2.waitKey(1)
	if k%256 == 32:
		print("predicting the output")
		test = gray
		predict = prediction(gray)

		idx = np.argmax(predict)
		pred = str(ALPHA[idx])
		test = cv2.putText(test, pred, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)

		# print(idx)
		cv2.imshow("prediction", test)
		pass
	elif k%256 == 27:
		print ('exit!')
		cv2.destroyAllWindows()
		sys.exit()


cam.release()							

