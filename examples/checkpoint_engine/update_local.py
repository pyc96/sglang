# torchrun --nproc-per-node 8 update_local.py --checkpoint-path /mnt/disks/datasets/models/MiniMax-M2 --inference-parallel-size 8 --sleep-time 60

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal, TypedDict
import zmq
import httpx
import torch
import torch.distributed as dist
from checkpoint_engine.ps import ParameterServer
from loguru import logger
from safetensors import safe_open
import threading


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype


@contextmanager
def timer(msg: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    logger.info(f"{msg} duration: {end - start:.2f} seconds")


def check_sglang_ready(
    endpoint: str, inference_parallel_size: int, uds: str | None = None
):
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    transport = None
    if uds is not None:
        transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                if retry_num % 10 == 0:
                    logger.warning(
                        f"fail to check sglang ready, retry {retry_num} times, error: {e}"
                    )
                retry_num += 1
                time.sleep(0.1)


def bind_zmq_socket(idx, total) -> tuple[zmq.Socket, list[tuple[str, str]]]:
    _zmq_ctx = zmq.Context()

    def zmq_handle(device_uuid: str) -> str:
        return f"ipc://@checkpoint-engine-{device_uuid}.sock"

    socket_paths = [(uid, zmq_handle(uid)) for uid in range(total)]
    socket = _zmq_ctx.socket(zmq.REQ)
    socket.bind(zmq_handle(idx))
    return socket, socket_paths


def req_inference(
    endpoint: str,
    inference_parallel_size: int,
    timeout: float = 300.0,
    uds: str | None = None,
    weight_version: str | None = None,
) -> Callable[[list[tuple[str, str]]], None]:
    src = 0

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            with httpx.Client(transport=httpx.HTTPTransport(uds=uds)) as client:
                resp = client.post(
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()

    return req_func


def split_tensors(
    checkpoint_path: str, rank: int, world_size: int
) -> dict[str, torch.Tensor]:
    index_fn = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_fn) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]
    weights_per_rank = (len(weight_map) + world_size - 1) // world_size
    fn_tensors: dict[str, list[str]] = defaultdict(list)
    weight_keys = list(weight_map.items())
    for name, file in weight_keys[
        rank * weights_per_rank : (rank + 1) * weights_per_rank
    ]:
        fn_tensors[file].append(name)
    named_tensors = {}
    for file, names in fn_tensors.items():
        with safe_open(os.path.join(checkpoint_path, file), framework="pt") as f:
            for name in names:
                named_tensors[name] = f.get_tensor(name)
    return named_tensors


def update_weights(
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    uds: str | None = None,
):
    check_sglang_ready(endpoint, inference_parallel_size, uds)
    dist.barrier()

    socket, socket_paths = bind_zmq_socket(rank, world_size)
    req_thread = threading.Thread(
        target=req_func,
        args=(socket_paths,),
    )
    req_thread.start()

    for name, tensor in named_tensors.items():
        handle = tensor.to(f"cuda:{rank}").untyped_storage()._share_cuda_()

        metadata = {
            "ipc_args": handle,
            "name": name,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
        }
        print(f"rank{rank}: sending weights")
        socket.send_pyobj(pickle.dumps(metadata))

        resp = socket.recv()
        if resp != b"":
            msg = resp.decode("utf-8")
            logger.error("fail fail")
        break
    print(f"rank{rank}: ending")
    socket.send_pyobj(None)
    print(f"rank{rank}: closing")
    socket.close()
    print(f"rank{rank}: joining")
    req_thread.join()
    print(f"rank: {rank}: return")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update weights example")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--sleep-time", type=int, default=0)
    parser.add_argument("--endpoint", type=str, default="http://localhost:7080")
    parser.add_argument("--inference-parallel-size", type=int, default=8)
    parser.add_argument("--uds", type=str, default=None)
    args = parser.parse_args()
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))

    # store = torch.distributed.TCPStore(
    #     os.getenv("MASTER_ADDR"),
    #     int(os.getenv("MASTER_PORT")),
    #     world_size,
    #     timeout=320,
    #     is_master=rank == 0,
    # )
    torch.distributed.init_process_group(
        "nccl",
        # store,
        world_size=world_size,
        rank=rank,
    )

    req_func = req_inference(
        args.endpoint,
        args.inference_parallel_size,
        uds=args.uds,
    )

    if os.path.exists(
        os.path.join(args.checkpoint_path, "model.safetensors.index.json")
    ):
        named_tensors = split_tensors(args.checkpoint_path, rank, world_size)

    update_weights(
        named_tensors,
        req_func,
        args.inference_parallel_size,
        args.endpoint,
        args.uds,
    )
    # time.sleep(args.sleep_time)
