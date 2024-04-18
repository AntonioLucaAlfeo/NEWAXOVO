import numpy as np

input_filename = "input.txt"
output_filename_1 = "output1.txt"
output_filename_2 = "output2.txt"

with open(input_filename, "r", encoding="utf-8") as file_object:

	data = file_object.read().split('\n')
	file_object.close()

print(data)
layer1_array = []
layer2_array = []

for layers in data:

	app = layers.strip("()").split(",")
	print(app)
	layer1_array.append(app[0])
	layer2_array.append(app[1])

print(layer1_array)
print(layer2_array)

with open(output_filename_1, "w", encoding="utf-8") as file_object:

	for number in layer1_array:

		file_object.write("%s\n" % number)
	
	file_object.close()

with open(output_filename_2, "w", encoding="utf-8") as file_object:

	for number in layer2_array:

		file_object.write("%s\n" % number)
	
	file_object.close()