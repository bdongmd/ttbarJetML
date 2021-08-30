import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from tqdm import tqdm 

def plotOutputScore(score, labels, output_dir='output'):
	outputScore = np.array(score)
	labels = np.array(labels)
	scanvalue = np.linspace(0.0, 1.0, num=100)
	cut = []
	eff = []
	purity = []
	signal = labels[labels==1]
	signal_score = score[labels==1]

	for i in tqdm (range(len(scanvalue)), desc="Calculating"):

		signal_tagged = signal[signal_score>scanvalue[i]]
		all_tagged = labels[score>scanvalue[i]]
		if len(all_tagged)==0:
			break
		cut.append(scanvalue[i])
		eff.append(len(signal_tagged)/len(signal))
		purity.append(len(signal_tagged)/len(all_tagged))

	print("========= plotting =========")
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(cut, eff, 'o', label='efficiency', color='black')
	plt.plot(cut, purity, 'o', label='purity', color='red')
	plt.xlabel('output score cut')
	plt.ylim((0,1.2))
	plt.legend()
	plt.savefig('{}/eff.pdf'.format(output_dir))

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.plot(eff, purity, 'o', color='black')
	plt.xlabel('efficiency')
	plt.ylabel('purity')
	plt.savefig('{}/eff_vs_purity.pdf'.format(output_dir))


inputfile = h5py.File(sys.argv[1], 'r')
labels = inputfile['labels'][:]
score = inputfile['outputScore'][:]
plotOutputScore(score, labels)
