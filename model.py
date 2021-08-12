from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Input, add
from keras.models import Model
from keras.optimizers import Adam
import sys

def private_DL1Model(InputShape, outputShape, h_layers, lr=0.01, drops=None, dropout=True):
	In = Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = Dense(h, kernel_initializer='glorot_uniform')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		if dropout:
			x = keras.layers.Dropout(drops[i])(x)

	if outputShape == 1:
		predictions = keras.layers.Dense(outputShape, activation='sigmoid')(x)
	elif outputShape == 2:
		predictions = keras.layers.Dense(outputShape, activation='softmax')(x)
	else:
		print("ERROR: wrong output numbers. The number of output categories can only be 1 or 2.")
		sys.exit()

	model = Model(inputs=In, outputs=predictions)

	model_optimizer = Adam(lr=lr)

	model.compile(
			loss = 'binary_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy'])

	return model
