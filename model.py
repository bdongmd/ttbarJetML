import tensorflow as tf
import tensorflow.keras as keras
import sys

def private_DL1Model(InputShape, outputShape, h_layers, lr=0.01, drops=None, dropout=True):
	In = keras.layers.Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = keras.layers.Dense(h, activation="relu",kernel_initializer='glorot_uniform')(x)
		if dropout:
			x = keras.layers.Dropout(drops[i])(x)
		x = keras.layers.BatchNormalization()(x)

	if outputShape == 1:
		predictions = keras.layers.Dense(outputShape, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
	elif outputShape == 2:
		predictions = keras.layers.Dense(outputShape, activation='softmax', kernel_initializer='glorot_uniform')(x)
	else:
		print("ERROR: wrong output numbers. The number of output categories can only be 1 or 2.")
		sys.exit()

	model = keras.models.Model(inputs=In, outputs=predictions)

	model_optimizer = keras.optimizers.Adam(lr=lr)

	model.compile(
			loss = 'categorical_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy'])

	return model
