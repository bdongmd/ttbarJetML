import numpy as np
import matplotlib.pyplot as plt
import sys

def plotAccLoss(trainInput, testInput, putVar, output_dir='models'):
	epochs = np.arange(1, len(trainLoss) + 1)
	
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(epochs, trainInput, 'o')
	plt.plot(epochs, testInput, 'o')
	if putVar == 'Loss':
		plt.legend(['training loss', 'testing loss'], loc='upper right')
		plt.ylabel('loss')
		plt.yscale('log')
	elif putVar == 'Acc':
		plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
		plt.ylabel('accuracy')
	else:
		print("ERROR: no such variables")
		sys.exit()

	plt.xlabel('epoch')
	#plt.ylim(0.5, 0.9)
	plt.savefig('{}/{}_compare.pdf'.format(output_dir, putVar))

def plotOutputScore(score, labels, output_dir='output'):
	outputScore = np.array(score)
	labels = np.array(labels)

	nbins=200
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.hist(outputScore[labels==1], bins=nbins, range=[0,1], density=True, label = 'signal', histtype='step')
	plt.hist(outputScore[labels==0], bins=nbins, range=[0,1], density=True, label = 'background', histtype='step')
	plt.ylabel('density')
	plt.xlabel('probability')
	plt.yscale('log')
	plt.legend()
	plt.savefig('{}/outputScore.pdf'.format(output_dir))
