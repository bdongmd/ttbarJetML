import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def variable_plotting(jets, outputFile = "output/test.pdf"):
	
	signal = jets.query('label==1')
	bkg = jets.query('label==0')

	nbins = 50
	with open("input_Variables.json") as vardict:
		variablelist = json.load(vardict)[:]

	varcounter = -1
	
	fig, ax = plt.subplots(10, 5, figsize=(25, 35))
	for i, axobjlist in enumerate(ax):
		for j, axobj in enumerate(axobjlist):
			varcounter+=1
			if varcounter < len(variablelist):
				var = variablelist[varcounter]
				
				b = bkg[var]
				s = signal[var]
				b.replace([np.inf, -np.inf], np.nan, inplace=True)
				s.replace([np.inf, -np.inf], np.nan, inplace=True)

				b = b.dropna()
				s = s.dropna()
			
				minval = np.amin(b)
				if 'pt' in var:
					maxval = np.percentile(u,99.99)
				else:
					maxval = max([np.amax(u), np.amax(c), np.amax(b)])*1.4
				binning = np.linspace(minval,maxval,nbins)
			
				axobj.hist(b, binning,histtype=u'step', color='orange',label='background',density=1)
				axobj.hist(s, binning,histtype=u'step', color='g',label='signal',density=1)
			
				axobj.legend()
				axobj.set_yscale('log',nonposy='clip')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')

	del signal, bkg
	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)

