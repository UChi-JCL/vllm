import torch
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Path to your cache
kv_path = Path('/local/kuntai/cache/kvcache')

# This will be our queue to hold tensors ready to be moved to GPU
tensor_queue = Queue()

def load_tensor(file_path):
    # Load the tensor from disk
    tensor = torch.load(file_path)
    # Put the loaded tensor into the queue
    tensor_queue.put(tensor)

def main():
    # Start time measurement
    st = time.time()

    # List all files in the directory
    files = list(kv_path.iterdir())
    files = files + files + files

    # Create a ThreadPoolExecutor for loading tensors
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all files to be loaded in parallel
        futures = [executor.submit(load_tensor, file) for file in files]

        # Wait for all tensors to be loaded and added to the queue
        for future in futures:
            future.result()

    # Now, consume the tensors from the queue and move them to the GPU
    while not tensor_queue.empty():
        tensor = tensor_queue.get()
        tensor.cuda()
        tensor_queue.task_done()

    print('CPU -> GPU time: ', time.time() - st)

if __name__ == "__main__":
    main()