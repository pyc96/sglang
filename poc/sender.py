import torch
import zmq
import pickle
from torch.multiprocessing.reductions import reduce_tensor


# Setup ZMQ
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://127.0.0.1:5555")

# 1. Load weights to GPU
# Important: weights must be a base tensor (not a view) for IPC
weights = torch.ones((4096, 4096)).cuda(1).half()

# ipc_handle = reduce_tensor(weights)
ipc_handle = weights.untyped_storage()._share_cuda_()

metadata = {
    "ipc_args": ipc_handle,
    "shape": list(weights.shape),
    "dtype": weights.dtype
}
socket.send(pickle.dumps(metadata))

print("Handle sent. Keeping process alive so memory remains valid...")
import time
time.sleep(60)
