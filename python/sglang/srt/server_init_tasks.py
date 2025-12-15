"""Server initialization tasks."""

import os
import asyncio
import time
import logging
import httpx
import subprocess

from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


async def parallel_weight_downloading_task(server_args: ServerArgs):
    """Run a command in a separate process at server startup."""
    model_path = server_args.parallel_weight_downloading_model_path
    last = model_path.removesuffix("/").split("/")[-1]
    local_path = "/tmp/model"
    command = f"gcloud storage cp --recursive {model_path} {local_path}" if model_path else None
    if not command:
        return

    logger.info(f"Executing server init task command: {command}")
    try:
        process = await asyncio.create_subprocess_shell(command)
        await process.wait()

        if process.returncode == 0:
            logger.info("Server init task completed successfully. Updating weights.")

            base_url = f"http://{server_args.host}:{server_args.port}"
            health_check_timeout = 60  # seconds
            start_time = time.time()
            server_is_healthy = False

            async with httpx.AsyncClient() as client:
                while time.time() - start_time < health_check_timeout:
                    try:
                        # Try to connect to the server
                        await client.get(base_url + "/health")
                        logger.info("Server is healthy. Proceeding to update weights.")
                        server_is_healthy = True
                        break
                    except httpx.ConnectError:
                        logger.info("Server not yet available, waiting...")
                    await asyncio.sleep(1)

                if server_is_healthy:
                    url = f"http://{server_args.host}:{server_args.port}/update_weights_from_disk"
                    payload = {
                        "model_path": f"{local_path}/{last}",
                        "load_format": "auto",
                        "abort_all_requests": True,
                        "is_async": False,
                        "torch_empty_cache": False,
                        "keep_pause": False,
                        "recapture_cuda_graph": False,
                        "token_step": 0,
                        "flush_cache": True,
                    }
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    logger.info("Successfully requested to update weights from disk.")
                else:
                    logger.error(f"Server health check failed after {health_check_timeout} seconds.")
        else:
            logger.error(
                f"Server init task command failed with return code {process.returncode}."
            )
    except Exception as e:
        logger.error(f"Failed to execute server init task command: {e}")