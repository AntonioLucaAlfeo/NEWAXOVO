import numpy as np

filename = "mean_and_std.txt"

mean_vector = np.empty((1, 12))
std_vector = np.empty((1, 12))

with open(filename, "r", encoding="utf-8") as file_object:

	data = file_object.read().replace('\t', '').split('\n')
	file_object.close()

for index, measurement in enumerate(data):

	a = measurement.replace(' ', '').split('±')
	#mean_vector[index%4, int(index/4)] = float(a[0])
	#std_vector[index%4, int(index/4)] = float(a[1])
	mean_vector[0, index] = float(a[0])
	std_vector[0, index] = float(a[1])

for index in np.arange(1):

	mean = np.mean(mean_vector[index, :])
	std = np.sqrt(np.mean(np.power(std_vector[index, :], 2)))
	print("%.4f ± %.4f" % (mean, std))

"""
with open(filename, "a", encoding="utf-8") as file_object:

	result = "\n %.4f ± %.4f" % (mean, std)
	file_object.write(result)
	file_object.close()
"""
