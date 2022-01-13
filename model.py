from keras.layers import BatchNormalization
from keras.layers import Dense, Activation, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from typing import Optional

def DL1Model(InputShape: int, h_layers: list, lr: float=0.01, drops: Optional[float]=None, dropout: bool=True):
	""" Define training model.
	Args:
	    InputShape (int): Number of input variables
	    h_layers (list): Number of nodes in each hidden layer
	    lr (float): learning rate
	    drops (Opitional[float]): Dropout probability in each hidden layer
	    dropout (bool): True to apply Dropout in each hidden layer, else not.
	"""
	In = Input(shape=[InputShape,])
	x = In
	for i, h in enumerate(h_layers[:]):
		x = Dense(h, kernel_initializer='glorot_uniform')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		if dropout:
			x = Dropout(drops[i])(x)

	predictions = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=In, outputs=predictions)

	model_optimizer = Adam(lr=lr)

	model.compile(
			loss = 'binary_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy'])

	return model
