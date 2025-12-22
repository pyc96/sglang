# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Checkpoint-engine integration for SGLang.
This module provides weight update functionality via IPC for checkpoint-engine compatibility.
"""
import logging
from typing import Callable, Dict, Optional

import gc
import torch
import zmq
import traceback
from typing import TypedDict
import pickle
from sglang.srt.model_loader.loader import device_loading_context

try:
    from checkpoint_engine.worker import update_weights_from_ipc
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )

logger = logging.getLogger(__name__)


class SGLangCheckpointEngineWorkerExtension:
    """
    Worker extension for SGLang to support checkpoint-engine IPC weight updates.
    This class provides the interface needed for checkpoint-engine integration.
    """

    def __init__(self):
        self._zmq_ctx: Optional[zmq.Context] = None

    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # We need to implement this to get the device UUID
        # This will be overridden when integrated into SGLang's worker
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_device_id(self) -> int:
        """Get the device ID."""
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        raise NotImplementedError(
            "This method should be overridden by SGLang integration"
        )

    def get_post_hook(self) -> Optional[Callable]:
        """Get the post-processing hook after weight loading."""
        return None

    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """
        Update weights from IPC communication.
        Args:
            zmq_handles: Dict mapping device UUID to ZMQ socket path
        """
        # here

        
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        # device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        rank = str(torch.distributed.get_rank())
        if rank not in zmq_handles:
            raise ValueError(
                f"Device UUID {rank} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        
        update_weights_from_ipc_local(
            self._zmq_ctx,
            zmq_handles[rank],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )



class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype



def _rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def _extract_weights(
    payload: list[FlattenedTensorMetadata], buffer: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    assert buffer is not None
    weights: list[tuple[str, torch.Tensor]] = []
    for item in payload:
        shape = item["shape"]
        if isinstance(shape, list | tuple):
            shape = torch.Size(shape)
        assert isinstance(shape, torch.Size)
        dtype = item["dtype"]
        tensor = buffer.view(dtype=dtype).view(shape)
        weights.append((item["name"], tensor))
    return weights


def update_weights_from_ipc_local(
    zmq_ctx: zmq.Context,
    zmq_handle: str,
    device_id: int,
    *,
    run: Callable[[list[tuple[str, torch.Tensor]]], None],
    post_hook: Callable[[], None] | None = None, 
):
    socket = zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handle)
    buffer: torch.Tensor | None = None

    # try:
    #     ipc_handle: tuple[Callable, tuple] = socket.recv_pyobj()
    #     assert isinstance(ipc_handle, tuple)
    #     buffer = _rebuild_ipc(ipc_handle, device_id)
    #     assert buffer.dtype == torch.uint8
    #     socket.send(b"")
    # except Exception as e:
    #     msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    #     socket.send_string(msg)
    #     socket.recv()  # wait for ack
    #     raise

    try:
        while True:
            # payload: dict | list[FlattenedTensorMetadata] | Exception | None = socket.recv_pyobj()
            payload: dict | None = socket.recv_pyobj()
            if payload is None:  # done signal
                print(f"rank: {torch.cuda.current_device()}: done!")
                # if post_hook is not None:
                #     post_hook()
                torch.cuda.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, bytes):  # still updating weights
                print(f"rank: {torch.cuda.current_device()}: got bytes!")
                meta = pickle.loads(payload)

                ipc_args = tuple(meta["ipc_args"])

                # This should now succeed without the "Cannot pickle" error
                shared_storage = torch.UntypedStorage._new_shared_cuda(*ipc_args)
                    
                # 2. Create a tensor view of that shared memory
                weight = torch.tensor(
                    [], dtype=meta["dtype"], device=f"cuda:{torch.cuda.current_device()}"
                ).set_(shared_storage).view(meta["shape"])
                weights = [(meta["name"], weight)]
                run(weights)
                torch.cuda.synchronize()
                socket.send(b"")

            # if isinstance(payload, list):  # still updating weights
            #     try:
            #         weights = _extract_weights(payload, buffer)
            #         run(weights)
            #         torch.cuda.synchronize()
            #         socket.send(b"")
            #     except Exception as e:  # noqa: BLE001
            #         # Send exception back to Parameter Server.
            #         # Don't raise here. Because all workers should quit in the same way by receiving the exception from PS
            #         # msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            #         # socket.send_string(msg)
            #         # Raise it anyway here first
            #         raise e
            elif isinstance(
                payload, Exception
            ):  # error occurred, got force quit signal from Parameter Server
                raise payload
            else:
                raise TypeError(f"Unexpected payload type: {type(payload)}")

    finally:
        print(f"rank: {torch.cuda.current_device()}: closing socket")
        socket.close()
        del buffer
        print(f"rank: {torch.cuda.current_device()}: gc")
        gc.collect()
        print(f"rank: {torch.cuda.current_device()}: empty cache")
        torch.cuda.empty_cache()
        print(f"rank: {torch.cuda.current_device()}: return")


class SGLangCheckpointEngineWorkerExtensionImpl(SGLangCheckpointEngineWorkerExtension):
    """
    Implementation of SGLangCheckpointEngineWorkerExtension that integrates with SGLang's model runner.
    This class provides the concrete implementation for checkpoint-engine IPC weight updates.
    """

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # Get device UUID for current device
        device_id = torch.cuda.current_device()
        try:
            return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
        except AssertionError as e:
            raise ValueError(f"Failed to get GPU UUID for device {device_id}") from e

    def get_device_id(self) -> int:
        """Get the device ID."""
        return torch.cuda.current_device()

    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        return self.model_runner.model.load_weights

    def get_post_hook(self) -> Optional[Callable]:
        """Get the post-processing hook after weight loading."""

        def post_hook():
            # Perform post-processing after weight loading similar to DefaultModelLoader
            try:
                from sglang.srt.model_loader.loader import device_loading_context

                # Process quantization methods after loading weights
                for _, module in self.model_runner.model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        # Move parameters to device if needed for quantization processing
                        target_device = torch.device(
                            "cuda", torch.cuda.current_device()
                        )
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)
                # Call model-specific post-loading hook if available
                if hasattr(self.model_runner.model, "post_load_weights"):
                    self.model_runner.model.post_load_weights()
            except Exception as e:
                logger.warning(f"Post-hook processing failed: {e}")

        return post_hook
