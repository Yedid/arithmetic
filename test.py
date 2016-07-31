import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def get_img(n, img_sz):
  im = np.zeros(img_sz)
  pil_im = Image.fromarray(im)
  draw = ImageDraw.Draw(pil_im)
  font = ImageFont.truetype("DejaVuSans.ttf", 12)
  sz = font.getsize(str(n))
  draw.text((img_sz[1]-sz[0], 2), str(n), font=font)
  im = np.asarray(pil_im)
  return im

def get_action(n1, n2):
  return n1 + n2


model = Sequential()
model.add(Flatten(input_shape=(15,60,2)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(900))
model.add(Activation('sigmoid'))

model.load_weights('model.h5')

model.compile(loss='mse', optimizer='adam')

NMax = 4999999
img_sz = (15,60)

data = np.zeros((img_sz[0], img_sz[1], 2))
mu = np.load('mu.npy')

for i in range(5):
	n1 = np.random.randint(0,NMax)
	n2 = np.random.randint(0,NMax)
	nres = get_action(n1,n2)
	data[:,:,0] = get_img(n1, img_sz)
	data[:,:,1] = get_img(n2, img_sz)
	plt.imshow(data[:,:,0], cmap='Greys_r')
	plt.title('First image.')
	plt.show()
	plt.imshow(data[:,:,1], cmap='Greys_r')
	plt.title('Second image.')
	plt.show()
	plt.imshow(get_img('?', img_sz), cmap='Greys_r')
	plt.title('The network ponders...')
	plt.show()
	im_pred = model.predict(data - mu)
	plt.imshow(im_pred.reshape((15,60)), cmap='Greys_r')
	plt.title('The prediction of the network:')
	plt.show() 
	im_res = get_img(nres, img_sz).reshape(img_sz)
	plt.imshow(im_res, cmap='Greys_r')
	plt.title('The correct response.')
	plt.show() 
	print np.mean((im_pred.flatten() - im_res.flatten())**2)
