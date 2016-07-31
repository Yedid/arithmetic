import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

batch_size = 100
nb_epoch = 250

def main():
	train_X = np.load('train_X.npy')
	train_y = np.load('train_y.npy')
	test_X = np.load('test_X.npy')
	test_y = np.load('test_y.npy')

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

	print model.summary()

	adam = Adam(0.001)
	#adagrad = Adagrad(lr=0.01)
	model.compile(loss='mse', optimizer=adam)

	model.fit(train_X, train_y, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=1, validation_data=(test_X, test_y))
	model.save_weights('model.h5', overwrite=True)

if __name__ == "__main__":
	main()