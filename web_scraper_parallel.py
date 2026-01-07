import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy.random as random
import pickle
from scipy.stats import norm

### send request to the web server and retrieve response
def submit_jobs(params, timeout=3600):
	url = "abaqus.oit.duke.edu:8000/submit"

	### number of jobs to submit
	batch_size = params.shape[0]

	### loop over jobs
	result_urls = [None]*batch_size
	for j in range(batch_size):

		### create a session to maintain cookies and session state
		with requests.Session() as session:

			### get the initial page to extract form data
			response = session.get(f"http://{url}")
			soup = BeautifulSoup(response.content, 'html.parser')

			### find all input fields
			input_fields = soup.find_all('input', class_='input')

			### update input fields with parameters
			input_fields[0]['value'] = 'dhd13'
			for i, param in enumerate(params[j][:], start=1):
				input_fields[i]['value'] = str(param)

			### submit the form
			response = session.post(f"http://{url}", data={field['name']: field['value'] for field in input_fields})

			### check if the response is a redirect
			if response.history:

				### get the redirected URL
				result_urls[j] = response.url

			### otherwise, display error
			else:
				print(response)
				raise RuntimeError("Failed to get redirected to results page")

	### return urls
	return result_urls

### wait in simulations to finish and fetch results
def get_results(result_urls,timeout=3600):

	### number of jobs submitted
	batch_size = len(result_urls)
	output_index = 3 # what is this?

	### loop over jobs
	output = []
	for j in range(batch_size):

		### create a session to maintain cookies and session state
		with requests.Session() as session:

			### poll the results page until the desired information is found or timeout is reached
			start_time = time.time()
			while True:
				if time.time() - start_time > timeout:
					print("Timeout reached. Unable to fetch results.")
					return None

				### fetch the results page
				connect_timeout = 10
				read_timeout = 10
				response = session.get(result_urls[j],timeout=(connect_timeout, read_timeout))
				soup = BeautifulSoup(response.content, 'html.parser')

				### find the table with id "results"
				results_table = soup.find('table', id='results')

				### check if the results table is found and contains data
				if results_table and len(results_table.find_all('tr')) > 1:

					### extract numerical values for the output
					output_all = []
					for row in results_table.find_all('tr')[1:]: # skip header row
						output_all.append(float(row.find('td').text))
					output.append(output_all[output_index])
					break

				### wait for a few seconds before polling again
				time.sleep(5)

	### return desired output
	return output


### adjust params array to keep imputs within bounds
def apply_bounds(X,mins,maxs):
	for p in range(len(X)):
		if X[p] < mins[p]:
			X[p] = mins[p]
		elif X[p] > maxs[p]:
			X[p] = maxs[p]
	return X

### create set of test points within bounds
def create_test_set(n_points_test,mins,maxs):
	X = np.zeros((n_points_test,len(mins)))
	for d in range(n_points_test):
		for p in range(len(mins)):
			X[d][p] = random.random()*(maxs[p]-mins[p]) + mins[p]
	return X

### acquisition function (EI)
def expected_improvement(X, gp, y_best):
    y_pred, y_std = gp.predict(X, return_std=True)
    z = (y_pred - y_best) / y_std
    ei = (y_pred - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

### maximize UCB from randomly sampled points
def calc_next_point(X_train,gp,y_best,n_points_test,mins,maxs):
	X_test = create_test_set(n_points_test,mins,maxs)
	ei = expected_improvement(X_test,gp,y_best)
	i_next = np.argmax(ei)
	X_next = apply_bounds(X_test[i_next][:],mins,maxs)
	return X_next

### estimate maximum of gaussian process
def calc_gp_max(gp,n_points_test,mins,maxs):
	X_test = create_test_set(n_points_test,mins,maxs)
	y_pred = gp.predict(X_test)
	y_best = max(y_pred)
	X_best = X_test[np.argmax(y_best)][:]
	return X_best,y_best

### round array to specified precisions
def apply_precision(X,precisions):
	for p in range(len(X)):
		X[p] = round(X[p],precisions[p])
	return X


### main function
def main():

	### search hyperparameter
	n_points_test = 10000
	n_batches = 1
	batch_size = 50

	### parameter bounds
	## params = [v_time, v_temp, x_s, x_cb, mix, grade]
	mins = [300,  403, 1, 1, 1,  0]
	maxs = [1000, 453, 10,  35, 11, 100]
	precisions = [0,0,0,0,0,0] # all integers

	### initial 5 points to initialize the Bayesian optimization
	X_train = np.array(\
			  [[520,   450,  7,  4,  4,  54],\
			   [874,   435,  9,  33,  8,  76],\
			   [991,   410,  5,  22,  9,  96],\
			   [627,  451,  2,  14,  11, 29],\
			   [301,  422,  4,  7,  1,  3]])

	### initial 5 outputs (sound dampening)
	y_train =  np.array(\
			   [0.1605, 0.0791, 0.0790, 0.0867, 0.1474])

	### initilize gaussian process
	gp = GaussianProcessRegressor()
	gp.fit(X_train,y_train)

	### find best validated result
	y_best = max(y_train)

	### optimization loop
	for b in range(n_batches):
		
		### calculate next batch of points
		X_next_batch = np.empty((0,len(mins)))
		for j in range(batch_size):

			### maximize ucb to find best next point
			X_next = calc_next_point(X_train,gp,y_best,n_points_test,mins,maxs)

			### round inputs to necessary precision
			X_next = apply_precision(X_next,precisions)

			### add point to batch list
			X_next_batch = np.vstack([X_next_batch,X_next])

		### run calculations with these points
		result_urls = submit_jobs(X_next_batch)

		### get results
		y_next = get_results(result_urls)

		### update training arrays
		X_train = np.vstack([X_train,X_next_batch])
		y_train = np.append(y_train,y_next)

		### find best validated result
		y_best = max(y_train)
		X_best = X_train[np.argmax(y_best)][:]

		### fit gaussian proccess to updated training arrays 
		gp.fit(X_train,y_train)

		### report progress
		X_gp_best,y_gp_best = calc_gp_max(gp,n_points_test,mins,maxs)
		print("Batch " + str(b+1) + " complete.")
		print("Best GP prediction: " + str(y_gp_best))
		print("Best validated result: " + str(y_best))
		print("Validated results:")
		print(y_train)
		print("")

		### save variables
		with open('./BO_vars_ei.pkl', 'wb') as f:
			pickle.dump([X_train,y_train,X_best,y_best], f)

main()

