# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
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
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
r"""Run local OASIS agents against a short-lived private Modal GPU server."""

import argparse
import asyncio
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import modal
from openai.types.chat import ChatCompletion

from examples.idea_propagation import run

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
FORWARD = """
import sys, urllib.request
request = urllib.request.Request(
    'http://127.0.0.1:8000/v1/chat/completions',
    data=sys.argv[1].encode(), headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=45) as response:
    print(response.read().decode())
"""


async def main(output, conditions, seeds):
    output.mkdir(parents=True, exist_ok=False)
    tags = {"idea-propagation-run": uuid.uuid4().hex}
    app = await modal.App.lookup.aio(
        "oasis-idea-propagation", create_if_missing=True
    )
    image = (
        modal.Image.from_registry("vllm/vllm-openai:v0.30.0")
        .entrypoint([])
        .run_commands("uv pip install --system zstandard==0.25.0")
        .run_commands(
            'python3 -c "from huggingface_hub import snapshot_download; '
            f"snapshot_download('{MODEL}', revision='{REVISION}', "
            "allow_patterns=['*.json','*.safetensors','*.txt','*.model'])\"",
        )
        .env({"OMP_NUM_THREADS": "1", "VLLM_NO_USAGE_STATS": "1"})
    )
    sandbox, started, runs = None, time.monotonic(), []
    try:
        sandbox = await modal.Sandbox.create.aio(
            "vllm",
            "serve",
            MODEL,
            "--revision",
            REVISION,
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.75",
            "--enforce-eager",
            "--generation-config",
            "vllm",
            app=app,
            image=image,
            tags=tags,
            gpu="L4",
            cpu=(4, 4),
            memory=(24576, 24576),
            timeout=900,
        )
        logger.info("Started bounded L4 sandbox: %s", sandbox.object_id)
        async with asyncio.timeout(450):
            while True:
                if await sandbox.poll.aio() is not None:
                    raise RuntimeError("vLLM exited before becoming ready.")
                check = await sandbox.exec.aio(
                    "python3",
                    "-c",
                    "import urllib.request; "
                    "urllib.request.urlopen('http://127.0.0.1:8000/health', "
                    "timeout=2)",
                    timeout=5,
                )
                await check.wait.aio()
                if check.returncode == 0:
                    break
                await asyncio.sleep(5)
        logger.info(
            "Real vLLM server is ready; starting %s pilot runs.",
            len(conditions) * len(seeds),
        )

        async def complete(**kwargs):
            # Forward only inference requests through authenticated Modal exec.
            # The model server has no public ports or persistent deployment.
            process = await sandbox.exec.aio(
                "python3", "-c", FORWARD, json.dumps(kwargs), timeout=60
            )
            stdout, stderr = await asyncio.gather(
                process.stdout.read.aio(), process.stderr.read.aio()
            )
            await process.wait.aio()
            if process.returncode:
                raise RuntimeError(
                    f"Private inference request failed: {stderr}"
                )
            return ChatCompletion.model_validate_json(stdout)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
        )
        for seed in seeds:
            for condition in conditions:
                args = SimpleNamespace(
                    smoke=False,
                    condition=condition,
                    seed=seed,
                    agents=4,
                    rounds=3,
                    model=MODEL,
                    base_url="http://127.0.0.1:8000/v1",
                    # Stop before a generated ChatML user/system turn.
                    stop=["<|im_start|>"],
                )
                directory = await run(args, async_client=client)
                destination = output / f"{condition}-seed-{seed}"
                shutil.copytree(directory, destination)
                summary = json.loads(
                    (destination / "summary.json").read_text()
                )
                runs.append(summary)
                logger.info(
                    "%s seed %s: %s calls, recipient whale posts=%s, "
                    "mentions after reset=%s",
                    condition,
                    seed,
                    summary["model_calls"],
                    summary["recipient_whale_post_authors"],
                    summary["recipient_whale_mentions_after_reset"],
                )
    finally:
        for leftover in [
            item
            async for item in modal.Sandbox.list.aio(
                app_id=app.app_id, tags=tags
            )
        ]:
            await leftover.terminate.aio()
            await leftover.wait.aio(raise_on_termination=False)
        if sandbox is not None:
            stdout, stderr = await asyncio.gather(
                sandbox.stdout.read.aio(), sandbox.stderr.read.aio()
            )
            (output / "server.stdout.log").write_text(stdout)
            (output / "server.stderr.log").write_text(stderr)
        remaining = [
            item
            async for item in modal.Sandbox.list.aio(
                app_id=app.app_id, tags=tags
            )
        ]
        report = {
            "model": MODEL,
            "decoding_stop": ["<|im_start|>"],
            "revision": REVISION,
            "vllm": "0.30.0",
            "gpu": "L4",
            "sandbox_timeout_seconds": 900,
            "cpu_cores_limit": 4,
            "memory_mib_limit": 24576,
            "elapsed_seconds_including_image_build": round(
                time.monotonic() - started, 2
            ),
            "completed_runs": len(runs),
            "expected_runs": len(conditions) * len(seeds),
            "active_sandboxes_after_cleanup": len(remaining),
            "runs": runs,
        }
        (output / "pilot.json").write_text(json.dumps(report, indent=2))
        logger.info("Saved pilot results to %s", output)
        if remaining:
            raise RuntimeError("Modal sandbox cleanup incomplete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=("unseeded", "preference", "relay")
    )
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    for name in (
        "social.agent",
        "social.twitter",
        "oasis.env",
        "camel.base_model",
        "camel.camel.agents.chat_agent",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    with modal.enable_output():
        asyncio.run(
            main(
                args.output_dir,
                (args.condition,)
                if args.condition
                else ("unseeded", "preference", "relay"),
                (args.seed,) if args.seed is not None else (0, 1),
            )
        )
