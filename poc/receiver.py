import torch
import zmq
import pickle

# Setup ZMQ
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://127.0.0.1:5555")

# The static buffer used by your CUDA Graph, initialized to zeros
static_buffer = torch.zeros((4096, 4096), device="cuda:1", dtype=torch.float16)
target = torch.ones((4096, 4096), device="cuda:1", dtype=torch.float16)

print("Capturing CUDA graph...")
# Capture a simple CUDA graph operation on the static_buffer
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    static_buffer.add_(1)
torch.cuda.current_stream().wait_stream(s)

with torch.cuda.graph(g):
    static_buffer.add_(1)

print("Server waiting for weights...")

msg = socket.recv()
meta = pickle.loads(msg)

# Convert list back to tuple
ipc_args = tuple(meta["ipc_args"])
print(ipc_args)

# This should now succeed without the "Cannot pickle" error
shared_storage = torch.UntypedStorage._new_shared_cuda(*ipc_args)
    
# 2. Create a tensor view of that shared memory
remote_weights = torch.tensor(
    [], dtype=meta["dtype"], device="cuda:1"
).set_(shared_storage).view(meta["shape"])

# 3. Perform the copy into the CUDA Graph buffer
# Since it's GPU-to-GPU on the same device, this is nearly instant
static_buffer.copy_(remote_weights)
torch.cuda.synchronize()

print("After copy, checking if static_buffer contains received weights (all ones):")
print((static_buffer == target).all())

print("\nReplaying CUDA graph (adding 1 to all elements)...")
g.replay()

print("After graph replay, checking if static_buffer is all twos:")
print((static_buffer == 2).all())
print("\nStatic weights updated via IPC and CUDA graph replayed successfully.")