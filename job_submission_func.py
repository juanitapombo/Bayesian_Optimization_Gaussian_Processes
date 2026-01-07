# %%

import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
## FUNCTIONS 

def submit_jobs(params, max_retries=3):
    url = "abaqus.oit.duke.edu:8000/submit"
    batch_size = params.shape[0]
    result_urls = [None] * batch_size

    for j in range(batch_size):
        retries = 0
        while retries < max_retries:
            with requests.Session() as session:
                response = session.get(f"http://{url}")
                soup = BeautifulSoup(response.content, 'html.parser')

                input_fields = soup.find_all('input', class_='input')

                input_fields[0]['value'] = 'jp630'
                for i, param in enumerate(params[j][:], start=1):
                    input_fields[i]['value'] = str(param)
                
                response = session.post(f"http://{url}", data={field['name']: field['value'] for field in input_fields})

                if response.history: # check if the response is a redirect
                    result_urls[j] = response.url
                    break  # exit the retry loop on success
                else:
                    print(response)
                    print(f"Attempt {retries + 1} failed for job {j}: {response}")
                    retries += 1

            if retries == max_retries:
                raise RuntimeError(f"Job {j} failed after {max_retries} attempts")

	# return job urls 
    return result_urls

def job_submission_loop(X):
    init_urls = []
    for i in range(0,len(X)):
        result_urls = submit_jobs(X[i].reshape(1, -1))
        print("Job submitted! Check results at:", result_urls[0])
        init_urls.append(result_urls[0])
    return init_urls

def get_results(urls, output_Optimization, output_list, timeout=3600):
    
	### number of jobs submitted
	batch_size = len(urls) 
	output_index = output_list.index(output_Optimization) ### index for desired output parameter for optimization 
      
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
				response = session.get(urls[j],timeout=(connect_timeout, read_timeout))
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

	return output

def create_candidate_points(n_points, param_mins, param_maxs):
    candidate_points = np.zeros((n_points, len(param_mins))) # initialize array of 0s with the desired shape
    # loop over params in X
    for i in range(len(param_mins)):
          # generate random ints within the bounds (exclusive)
          candidate_points[:, i] = np.random.randint(param_mins[i], param_maxs[i], size=n_points)
    return candidate_points  

def expected_improvement(X, gp, y_best, maximization=True):
    y_pred, y_std = gp.predict(X, return_std=True)

    if (maximization):
        z = (y_pred - y_best) / y_std   
        ei = (y_pred - y_best) * norm.cdf(z) + y_std * norm.pdf(z)
    else: # minimization
        z = (y_best - y_pred) / y_std   
        ei = (y_best - y_pred) * norm.cdf(z) + y_std * norm.pdf(z)

    return ei

def optimization_step(n_candidate_pts, param_mins, param_maxs, batchsize, maximization=True):

    X_cand = create_candidate_points(n_candidate_pts, param_mins, param_maxs)
    X_cand_ei = expected_improvement(X_cand, gp, y_best, maximization=True)
    if (maximization):
        selected_ids = np.argsort(X_cand_ei)[-batchsize:]   # Get indices of the max X candidates
    else:
        selected_ids = np.argsort(X_cand_ei)[:batchsize]    # Get indices of the min X candidates
    X_next = X_cand[selected_ids]

    return X_next 

def plot_evolutions(y_best_history, optimization_param, n_batches, X_train, X_next):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot y_best over iterations
    axes[0].plot(range(0, len(y_best_history)), y_best_history, marker='o', linestyle='-', color='k')
    axes[0].set_xlabel("Batch Number", fontsize=14)
    axes[0].set_ylabel(f"y_best for {optimization_param}", fontsize=14)
    axes[0].set_xticks(range(0, n_batches + 1)) 
    axes[0].set_xlim(-0.2, n_batches + 0.2)
    axes[0].set_title(f"Evolution of Y_best for {optimization_param}", fontsize=14)
    axes[0].tick_params(axis='both', labelsize=14)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c='gray', edgecolors='black', label="All points")
    X_next_pca = pca.transform(X_next)  
    axes[1].scatter(X_next_pca[:, 0], X_next_pca[:, 1], c='red', edgecolors='black', label="New points")
    axes[1].set_xlabel("First Principal Component", fontsize=14)
    axes[1].set_ylabel("Second Principal Component", fontsize=14)
    axes[1].set_title("Evolution of X_train (PCA onto 2 PCs)", fontsize=14)
    axes[1].legend(frameon=False, fontsize=14)
    axes[1].legend(frameon=False, fontsize=14, loc='upper left', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.show()

def GP_initialization(X_train, optimization_param, output_list):
    all_urls = []
    init_urls = job_submission_loop(X_train)
    all_urls.append(init_urls)
    
    y_train = get_results(init_urls, optimization_param, output_list, timeout=3600)
     
    return all_urls, y_train

def GP_workflow(X_train, y_train, n_batches, batchsize, n_candidate_pts, param_mins, param_maxs, output_list, all_urls, maximization=True):

    # GP Initialization Stage
    gp = GaussianProcessRegressor()
    gp.fit(X_train,y_train)                                     # initilize gaussian process
    y_best = max(y_train) if maximization else min(y_train)     # Update best known value

    # store history 
    y_best_history = [y_best]

    for batch in range(n_batches):
        print(f"Batch {batch + 1}/{n_batches}")

        # Select new candidate points
        X_next = optimization_step(n_candidate_pts, param_mins, param_maxs, batchsize, maximization=True)

        # Submit jobs and retrieve results
        result_urls = job_submission_loop(X_next)
        all_urls.append(result_urls)
        y_next = get_results(result_urls, 'sound_damping', output_list, timeout=3600)

        # Update training data
        X_train = np.vstack([X_train,X_next])
        y_train = np.append(y_train,y_next)
    
        # Retrain GP on updated data
        gp.fit(X_train, y_train)                                

        # Update best known value
        y_best = max(y_train) if maximization else min(y_train) 
        y_best_history.append(y_best)

        # Plot the evolution of y_best and X_train points onto the first 2 principal component space
        plot_evolutions(y_best_history, optimization_param, n_batches, X_train, X_next)

    return all_urls, y_best_history, X_train


# %%
# Job pecific Inputs
n_batches = 10
n_candidate_pts = 100000  
batchsize = 5   
param_mins = [300,  403, 1, 1, 1,  0]
param_maxs = [1000, 453, 10,  35, 11, 100]
output_list = ['material_cost','heating_cost','mixing_cost','sound_damping','rolling_resistance','abrasion','dry_grip','wet_grip']

# Selection of Optimization Parameter
optimization_param = 'sound_damping'
maximization=True

# X Params for the 5 Points for GP Initialization
X_train = np.array(\
            [[520,   450,  7,  4,  4,  54],\
            [874,   435,  9,  33,  8,  76],\
            [991,   410,  5,  22,  9,  96],\
            [627,  451,  2,  14,  11, 29],\
            [301,  422,  4,  7,  1,  3]])  

# GP Initialization
all_urls, y_train = GP_initialization(X_train, optimization_param, output_list)
     
# GP Workflow
all_urls, y_best_history, X_train = GP_workflow(X_train, y_train, n_batches, batchsize, n_candidate_pts, param_mins, param_maxs, output_list, all_urls, maximization=True)





# %%
