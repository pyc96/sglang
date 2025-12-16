"""Launch the inference server."""

import asyncio
import os
import multiprocessing
import sys
from sglang.srt.server_init_tasks import parallel_weight_downloading_task

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree


def run_server(server_args):
    """Run the server based on server_args.grpc_mode."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    else:
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


def run_init_task(server_args):
    """Wrapper to run the async init task."""
    asyncio.run(parallel_weight_downloading_task(server_args))


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])
    if server_args.parallel_weight_downloading_model_path:
        init_proc = multiprocessing.Process(target=run_init_task, args=(server_args,))
        init_proc.start()

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
