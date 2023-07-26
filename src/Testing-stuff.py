import concurrent.futures
from tqdm import tqdm

colours = ["#AAFFAA", "#FF8888", "#9999FF", "#BBBBBB"]

def dostuff(arg):
    import time
    import random
    random.seed(arg)
    time.sleep(random.random()*2)

# Replace your_compute_function with the actual computation function you are running
def your_compute_function(prox_id, arg, pbar, worker_id):
    # Simulate some computation
    import time
    # pbar.colour = colours[worker_id]
    # from tqdm.notebook import tqdm
    # print("you gave me this arg: ", arg)
    # with tqdm(total=10, desc=f'{prox_id}', leave=True, ncols=10) as pbar:
    for i in range(10):
        # time.sleep(0.1)
        import time
        import random
        random.seed(arg)
        time.sleep(random.random())
        pbar.update()
    return arg * arg

def process_task(prox_id, arg):
    # Call your computation function
    return your_compute_function(prox_id, arg)

if __name__ == "__main__":
    # Number of processes you want to run in parallel
    num_processes = 4

    # List of input arguments for your computation function
    input_args = list(range(1, 11))
    pbars = {
        prox_id : tqdm(total=10, desc=f'{prox_id:2}', leave=True, colour="green") for prox_id in input_args
    }

    # Initialize the progress bar
    # with tqdm(total=len(input_args)) as pbar:
    # pbar = tqdm(total=len(input_args))
    # Using ThreadPoolExecutor for thread-based parallelism
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_processes)
    # Submit the tasks to the executor
    future_to_arg = {executor.submit(your_compute_function, arg, arg, pbars[arg], arg%num_processes): arg for arg in input_args}

    # Process the completed tasks as they finish
    for future in concurrent.futures.as_completed(future_to_arg):
        result = future.result()
        # pbar.update()
        # You can collect the results if needed
        # results.append(result)

    # No need to close or join anything for ThreadPoolExecutor
    # Now you have the results of your computation in the 'results' list
    # print("Results:", results)
