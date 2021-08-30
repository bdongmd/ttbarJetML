import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

def plotOutputScore(score, labels, output_dir='output'):
	outputScore = np.array(score)
	labels = np.array(labels)
	scanvalue = np.linspace(0.0, 1.0, num=100)
	eff = []
	purity = []
	signal = labels[labels==1]
	signal_score = score[labels==1]

	for i in range(len(scanvalue)):
		signal_tagged = signal[signal_score>scanvalue[i]]
		all_tagged = labels[score>scanvalue[i]]
		eff.append(len(signal_tagged)/len(signal))
		purity.append(len(signal_tagged)/len(all_tagged))

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(scanvalue, eff, 'o', label='efficiency')
	plt.plot(scanvalue, purity, 'o', label='purity')
	plt.xlabel('output score cut')
	plt.ylabel('efficiency/purity (%)')
	plt.ylim((0,120))
	plt.legend()
	plt.savefig('{}/eff.pdf'.format(output_dir))

inputfile = h5py.File(sys.argv[1], 'r')
labels = inputfile['labels'][:]
score = inputfile['outputScore'][:]
plotOutputScore(score, labels)
