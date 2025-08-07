import psutil
import time

def get_system_info():
    """
    Gets system hardware information.

    Returns:
        dict: A dictionary containing system information like RAM and CPU cores.
    """
    return {
        "ram": psutil.virtual_memory().total / (1024**3),  # in GB
        "cpu_cores": psutil.cpu_count(logical=False),
    }

def estimate_training_time(model, data_size):
    """
    A basic function to estimate the training time for a given model and data size.
    This is a placeholder and can be replaced with a more sophisticated model.

    Args:
        model: The model to be trained.
        data_size (int): The size of the training data.

    Returns:
        float: Estimated training time in seconds.
    """
    # Simple estimation based on model type and data size
    model_type = type(model).__name__
    if "RandomForest" in model_type or "GradientBoosting" in model_type:
        return data_size * 0.0001
    elif "SVC" in model_type:
        return data_size * 0.0005
    else:
        return data_size * 0.0002

def monitor_resources(func):
    """
    A decorator to monitor resource usage (CPU, memory) of a function.
    """
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        
        # Get resource usage before execution
        mem_before = process.memory_info().rss / 1024**2  # in MB
        cpu_before = process.cpu_percent(interval=None)

        result = func(*args, **kwargs)

        # Get resource usage after execution
        mem_after = process.memory_info().rss / 1024**2  # in MB
        cpu_after = process.cpu_percent(interval=None)
        end_time = time.time()

        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f}s")
        print(f"Memory usage: {mem_after - mem_before:.2f} MB")
        print(f"CPU usage: {cpu_after - cpu_before:.2f}%")
        
        return result
    return wrapper