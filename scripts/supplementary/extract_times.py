from __future__ import division

import os

import numpy as np
import pandas as pd
import seaborn as sns

name_cpu = "palladio_session_NOGPU"
name_gpu = "palladio_session_GPU"

def main():

	times_cpu = dict()
	times_gpu = dict()

	for folder in os.listdir('.'):
		if (folder.startswith("multivariate")):

			foldername_arr = folder.split('_')

			n = foldername_arr[2]
			p = foldername_arr[4]

			report_cpu = os.path.join(folder, name_cpu, 'report.txt')
			report_gpu = os.path.join(folder, name_gpu, 'report.txt')

			with open(report_cpu, 'r') as f_cpu, open(report_gpu, 'r') as f_gpu:

				### READ TIME FOR CPU
				ss = f_cpu.readline()
				ss = ss[len("Total elapsed time: "):] ### only the time

				time_arr = ss.split(':')
				total_time_cpu = int(time_arr[0]) * 60 * 60 + int(time_arr[1]) * 60 + int(time_arr[2])

				if not n in times_cpu.keys():
					times_cpu[n] = dict()

				times_cpu[n][p] = total_time_cpu

				### READ TIME FOR GPU
				ss = f_gpu.readline()
				ss = ss[len("Total elapsed time: "):] ### only the time

				time_arr = ss.split(':')
				total_time_gpu = int(time_arr[0]) * 60 * 60 + int(time_arr[1]) * 60 + int(time_arr[2])

				if not n in times_gpu.keys():
					times_gpu[n] = dict()

				times_gpu[n][p] = total_time_gpu

				# print total_time/3600


	ns = times_cpu.keys()
	ps = times_cpu[ns[0]].keys()

	ns = [int(x) for x in ns]
	ps = [int(x) for x in ps]

	ns = sorted(ns)
	ps = sorted(ps)

	times_cpu_arr = np.empty((len(ns), len(ps)))
	times_gpu_arr = np.empty((len(ns), len(ps)))

	for i, n in enumerate(ns):
		for j, p in enumerate(ps):

			times_cpu_arr[i,j] = times_cpu[str(n)][str(p)] ### BLEAH!
			times_gpu_arr[i,j] = times_gpu[str(n)][str(p)] ### BLEAH!


	cmap = ["RdBu", "PRGn", "PiYG", "BrBG", "RdYlGn", "coolwarm"]

	speedup_arr = times_cpu_arr/times_gpu_arr
	df = pd.DataFrame(data=speedup_arr, index=ns, columns=ps)
	sns.set_context("notebook")
	sns.heatmap(df, cmap=cmap[0], center=1, annot=True, fmt='1.2f')
	sns.plt.title('GPU Speedup')
	sns.plt.ylabel(r'Number of samples $n$')
	sns.plt.xlabel(r'Number of dimensions $d$')
	sns.plt.show()



	print speedup_arr



if __name__ == '__main__':
	main()
