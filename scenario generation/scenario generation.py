import numpy as np
from pulp import *
import random
import math
from scipy.stats import norm

def inverse_cdf_normal(mean, std_dev, p):
    return norm.ppf(p, loc=mean, scale=std_dev)

n = 100
miu_wheat, sigma_wheat = 2.5, 0.5
miu_corn, sigma_corn = 3, 0.6
miu_sugar, sigma_sugar = 20, 4
num_of_kisi = 3

random_list = np.zeros(n - 1)
scenarios = np.zeros((n, num_of_kisi))
probability = np.zeros(n)

# Generating random numbers for n scenarios
for i in range(n - 1):
	random_list[i] = random.random()

sorted_numbers = np.sort(random_list)

# Generating scenarios from randomNumbers and assign its probability
for i, prob in enumerate(sorted_numbers):
	if i == 0:
		probability[i] = prob
		scenarios[i][0] = inverse_cdf_normal(mean=miu_wheat, std_dev=sigma_wheat, p=prob/2)
		scenarios[i][1] = inverse_cdf_normal(mean=miu_corn, std_dev=sigma_corn, p=prob/2)
		scenarios[i][2] = inverse_cdf_normal(mean=miu_sugar, std_dev=sigma_sugar, p=prob/2)
  
	elif i == n - 2:
		probability[i] = (sorted_numbers[i] - sorted_numbers[i - 1])
		central_prob = (sorted_numbers[i] + sorted_numbers[i - 1]) / 2
		scenarios[i][0] = inverse_cdf_normal(mean=miu_wheat, std_dev=sigma_wheat, p=central_prob)
		scenarios[i][1] = inverse_cdf_normal(mean=miu_corn, std_dev=sigma_corn, p=central_prob)
		scenarios[i][2] = inverse_cdf_normal(mean=miu_sugar, std_dev=sigma_sugar, p=central_prob)
		probability[n - 1] = (1 - sorted_numbers[n - 2])
		central_prob = (1 + sorted_numbers[n - 2]) / 2
		scenarios[n - 1][0] = inverse_cdf_normal(mean=miu_wheat, std_dev=sigma_wheat, p=central_prob)
		scenarios[n - 1][1] = inverse_cdf_normal(mean=miu_corn, std_dev=sigma_corn, p=central_prob)
		scenarios[n - 1][2] = inverse_cdf_normal(mean=miu_sugar, std_dev=sigma_sugar, p=central_prob)
  
	else:
		probability[i] = (sorted_numbers[i] - sorted_numbers[i - 1])
		central_prob = (sorted_numbers[i] + sorted_numbers[i - 1]) / 2
		scenarios[i][0] = inverse_cdf_normal(mean=miu_wheat, std_dev=sigma_wheat, p=central_prob)
		scenarios[i][1] = inverse_cdf_normal(mean=miu_corn, std_dev=sigma_corn, p=central_prob)
		scenarios[i][2] = inverse_cdf_normal(mean=miu_sugar, std_dev=sigma_sugar, p=central_prob)
		
total_scenario, total_probabilities = scenarios, probability

# Creating the index matrices for each scenario
total_index = np.zeros(n)
for i, scenario in enumerate(total_scenario):
    total_index[i] = i    
total_index

first_year = {"removed_scenarios": {"scenario": [], "probability": [], "index": []},
              "keeped_scenarios": {"scenario": total_scenario, "probability": total_probabilities, "index": total_index}}

# Define distance matrix to store distance between scenario i & j
minimum_distances = []
distance_matrix = np.zeros((n, n))

# Calculating each indexes of Distance Matrix
for i1, i in enumerate(first_year["keeped_scenarios"]["scenario"]):
	for i2, j in enumerate(first_year["keeped_scenarios"]["scenario"]):
		if i1 != i2:
			# Calcualte distances based on Euclidean Distance
			distance_matrix[i1][i2] = (math.sqrt(math.pow((i[0] - j[0]), 2) + math.pow((i[1] - j[1]), 2) + math.pow((i[2] - j[2]), 2)))
		else:
			distance_matrix[i1][i2] = 10000
   
# print(distance_matrix)

# find smallest (Probability * distance)
for i, distance in enumerate(distance_matrix):
	# Calculating (probability * distance) for each scenario
	minimum_distances.append(first_year["keeped_scenarios"]["probability"][i] * min(distance))

min_value = min(minimum_distances)
min_index = minimum_distances.index(min_value)
first_year["removed_scenarios"]["scenario"].append(first_year["keeped_scenarios"]["scenario"][min_index])
first_year["removed_scenarios"]["probability"].append(first_year["keeped_scenarios"]["probability"][min_index])
first_year["removed_scenarios"]["index"].append(first_year["keeped_scenarios"]["index"][min_index])
first_year["keeped_scenarios"]["scenario"] = np.delete(first_year["keeped_scenarios"]["scenario"], min_index, axis=0)
first_year["keeped_scenarios"]["probability"] = np.delete(first_year["keeped_scenarios"]["probability"], min_index, axis=0)
first_year["keeped_scenarios"]["index"] = np.delete(first_year["keeped_scenarios"]["index"], min_index)

# Define while condition
stop_condition = False
minimum_distances = []

# Stop Condition: # of keeped scenarios <= 15
while not stop_condition:
	total_distances = []
	# Candidate the keeped scenarios
	for i, j in enumerate(first_year["keeped_scenarios"]["scenario"]):
		first_year["removed_scenarios"]["scenario"].append(j)
		first_year["removed_scenarios"]["probability"].append(first_year["keeped_scenarios"]["probability"][i])
		first_year["removed_scenarios"]["index"].append(first_year["keeped_scenarios"]["index"][i])
		np.delete(first_year["keeped_scenarios"]["scenario"], i, 0)
		np.delete(first_year["keeped_scenarios"]["probability"], i, 0)
		keeped_index = np.delete(first_year["keeped_scenarios"]["index"], i, 0)
		# print(keeped_index)
		# Calculating the distance between keeped and removed scenarios
		for i1, k in enumerate(first_year["removed_scenarios"]["index"]):
			distances = []
			for i2, z in enumerate(keeped_index):
				distances.append(distance_matrix[int(k)][int(z)])
			minimum_distances.append(first_year["removed_scenarios"]["probability"][i1] * min(distances))
		# print(minimum_distances)
		total_distances.append(sum(minimum_distances))
		# print(total_distances)
		first_year["removed_scenarios"]["scenario"].pop(-1)
		first_year["removed_scenarios"]["probability"].pop(-1)
		first_year["removed_scenarios"]["index"].pop(-1)
		minimum_distances = []
	# Find the best choice for removing
	min_value = min(total_distances)
	min_index = total_distances.index(min_value)

	# Check the stop condition
	if len(first_year["keeped_scenarios"]["scenario"]) <= 40:
		stop_condition = True
	else:
		first_year["removed_scenarios"]["scenario"].append(first_year["keeped_scenarios"]["scenario"][min_index])
		first_year["removed_scenarios"]["probability"].append(first_year["keeped_scenarios"]["probability"][min_index])
		first_year["removed_scenarios"]["index"].append(first_year["keeped_scenarios"]["index"][min_index])
		first_year["keeped_scenarios"]["scenario"] = np.delete(first_year["keeped_scenarios"]["scenario"], min_index, 0)
		first_year["keeped_scenarios"]["probability"] = np.delete(first_year["keeped_scenarios"]["probability"], min_index, 0)
		first_year["keeped_scenarios"]["index"] = np.delete(first_year["keeped_scenarios"]["index"], min_index, 0)
		

print(f"Sum of generated scenarios before clustering: {sum(first_year['keeped_scenarios']['probability'])}")

# create dictionary to store central represenatative and its probability
stage_cluster = {"cluster": first_year["keeped_scenarios"]["scenario"],
					"probability": first_year["keeped_scenarios"]["probability"]}

# Find the best represetative for removed scenario
for i1, j in enumerate(first_year["removed_scenarios"]["scenario"]):
	distances = []
	for i2, k in enumerate(first_year["keeped_scenarios"]["scenario"]):
		distance = math.sqrt(math.pow((j[0] - k[0]), 2) + math.pow((j[1] - k[1]), 2) + math.pow((j[2] - k[2]), 2))
		distances.append(distance)
	min_value = min(distances)
	min_index = distances.index(min_value)
	# modifying the probability of each cluster
	stage_cluster["probability"][min_index] += first_year["removed_scenarios"]["probability"][i1]
	
print(f"Sum of generated scenarios after clustering and modification: {sum(first_year['keeped_scenarios']['probability'])}")

# print final scenarios
print(first_year['keeped_scenarios'])