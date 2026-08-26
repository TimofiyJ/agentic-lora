"""Automated vast.ai deployment script for the LoRA orchestration stack.

Usage:
    python scripts/deploy.py [--keep] [--instance-id ID] [--gpu-ram N] [--disk N]

Flags:
    --keep          Skip teardown after test (default: destroy instance)
    --instance-id   Skip creation, use existing running instance
    --gpu-ram N     Minimum VRAM in GB (default: 24)
    --disk N        Disk size in GB (default: 60)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).parent.parent


def run(cmd: str) -> str:
    """Run a shell command, print it, and return stdout.

    Args:
        cmd: Shell command string to execute.

    Returns:
        stdout of the completed process as a string.

    Raises:
        subprocess.CalledProcessError: If the command exits with non-zero status.
    """
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}", file=sys.stderr)
        result.check_returncode()
    return result.stdout.strip()


def vastai(*args: str, raw: bool = True, parse: bool = True) -> dict | list | str:
    """Call the vastai CLI and optionally return parsed JSON output.

    Args:
        *args: Arguments passed to the vastai command.
        raw: Whether to append --raw for JSON output (default True).
        parse: Whether to parse stdout as JSON (default True). Set False
            for commands that return plain text or empty output.

    Returns:
        Parsed JSON as dict or list when parse=True, else raw stdout string.

    Raises:
        subprocess.CalledProcessError: If vastai exits with non-zero status.
        json.JSONDecodeError: If parse=True and response is not valid JSON.
    """
    flags = " --raw" if raw else ""
    cmd = f"vastai {' '.join(args)}{flags}"
    output = run(cmd)
    return json.loads(output) if parse else output


def find_offer(gpu_ram: int, disk: int) -> int:
    """Search for the cheapest rentable GPU offer meeting requirements.

    Args:
        gpu_ram: Minimum GPU VRAM in GB.
        disk: Minimum disk space in GB.

    Returns:
        Offer ID of the cheapest qualifying instance.

    Raises:
        RuntimeError: If no matching offers are found.
    """
    query = f"gpu_ram>={gpu_ram} disk_space>={disk} cuda_vers>=12.1 rentable=true"
    offers = vastai("search", "offers", f"'{query}'", "--order", "dph_total")
    if not offers:
        raise RuntimeError(f"No offers found matching: {query}")
    offer = offers[0]
    print(f"  Found offer {offer['id']} — ${offer['dph_total']:.3f}/hr, "
          f"{offer['gpu_name']} {offer['gpu_ram']}GB VRAM")
    return offer["id"]


def create_instance(offer_id: int, disk: int) -> int:
    """Rent a vast.ai instance from an offer.

    Args:
        offer_id: The offer ID to rent.
        disk: Disk size in GB.

    Returns:
        The new instance (contract) ID.

    Raises:
        RuntimeError: If instance creation fails or returns no ID.
    """
    result = vastai(
        "create", "instance", str(offer_id),
        "--image", "pytorch/pytorch:@vastai-automatic-tag",
        "--disk", str(disk),
        "--ssh", "--direct",
    )
    instance_id = result.get("new_contract")
    if not instance_id:
        raise RuntimeError(f"Instance creation failed: {result}")
    print(f"  Created instance {instance_id}")
    return instance_id


def wait_until_running(instance_id: int, timeout: int = 1200) -> None:
    """Poll until the instance reaches 'running' status.

    Args:
        instance_id: The instance ID to monitor.
        timeout: Maximum seconds to wait before raising.

    Raises:
        TimeoutError: If instance is not running within timeout.
    """
    deadline = time.time() + timeout
    terminal_failures = {"exited", "unknown", "offline"}
    while time.time() < deadline:
        info = vastai("show", "instance", str(instance_id))
        status = info.get("actual_status", "")
        print(f"  Status: {status}")
        if status == "running":
            return
        if status in terminal_failures:
            raise RuntimeError(f"Instance {instance_id} reached unrecoverable status: {status}")
        time.sleep(10)
    raise TimeoutError(f"Instance {instance_id} did not reach 'running' within {timeout}s")


def find_ssh_key() -> str | None:
    """Return the path to the first available local SSH private key.

    Returns:
        Absolute path string, or None if no key is found.
    """
    for candidate in ("~/.ssh/id_ed25519", "~/.ssh/id_rsa"):
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return None


def copy_code(instance_id: int) -> None:
    """Copy source code to the remote instance via rsync over SSH.

    Uses rsync directly with the instance SSH connection details so that
    vastai SSH key registration is not required.

    Args:
        instance_id: The target instance ID.
    """
    info = vastai("show", "instance", str(instance_id))
    host = info["ssh_host"]
    port = info["ssh_port"]

    key_flag = ""
    key = find_ssh_key()
    if key:
        key_flag = f" -i {key}"

    ssh_cmd = f"ssh -p {port}{key_flag} -o StrictHostKeyChecking=no"

    pairs = [
        (REPO_ROOT / "src" / "lora_controller", "root@" + host + ":/root/lora_controller/"),
        (REPO_ROOT / "src" / "orchestrator", "root@" + host + ":/root/orchestrator/"),
        (REPO_ROOT / "adapters.yaml", "root@" + host + ":/root/adapters.yaml"),
    ]
    for local, remote in pairs:
        run(f'rsync -az -e "{ssh_cmd}" {local} {remote}')


def setup_server(instance_id: int, hf_token: str) -> None:
    """Install dependencies and write environment config on the instance.

    Args:
        instance_id: The target instance ID.
        hf_token: HuggingFace API token for model downloads.
    """
    cmds = [
        "python3 -m venv ~/venv",
        "~/venv/bin/pip install -q vllm",
        "~/venv/bin/pip install -q fastapi uvicorn httpx pyyaml python-dotenv rouge-score",
        f"echo 'HF_TOKEN={hf_token}' > ~/.env",
    ]
    for cmd in cmds:
        vastai("execute", str(instance_id), f'"{cmd}"', parse=False)


def start_services(instance_id: int) -> None:
    """Launch vLLM, controller, and orchestrator as background processes.

    Args:
        instance_id: The target instance ID.
    """
    vllm_cmd = (
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True "
        "HUGGING_FACE_HUB_TOKEN=$(grep HF_TOKEN ~/.env | cut -d= -f2) "
        "nohup ~/venv/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct "
        "--enable-lora --max-loras 4 --max-lora-rank 256 "
        "--max-model-len 65536 --port 8000 > ~/vllm.log 2>&1 &"
    )
    controller_cmd = (
        "ADAPTERS_CONFIG=~/adapters.yaml VLLM_TIMEOUT=300 "
        "nohup ~/venv/bin/uvicorn lora_controller.app:app "
        "--port 8001 --host 0.0.0.0 > ~/controller.log 2>&1 &"
    )
    orchestrator_cmd = (
        "PLANNER_MODEL=meta-llama/Llama-3.1-8B-Instruct "
        "nohup ~/venv/bin/uvicorn orchestrator.app:app "
        "--port 8002 --host 0.0.0.0 > ~/orchestrator.log 2>&1 &"
    )
    for cmd in (vllm_cmd, controller_cmd, orchestrator_cmd):
        vastai("execute", str(instance_id), f'"{cmd}"', parse=False)


def wait_healthy(instance_id: int, timeout: int = 600) -> None:
    """Poll the controller health endpoint until vLLM is reachable.

    Args:
        instance_id: The target instance ID.
        timeout: Maximum seconds to wait.

    Raises:
        TimeoutError: If services are not healthy within timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = vastai(
                "execute", str(instance_id),
                '"curl -s http://localhost:8001/health"',
                raw=False,
            )
            health = json.loads(result) if isinstance(result, str) else result
            if health.get("vllm_reachable"):
                print("  All services healthy.")
                return
            print(f"  Health: {health}")
        except (json.JSONDecodeError, subprocess.CalledProcessError) as e:
            print(f"  Not ready yet: {e}")
        time.sleep(15)
    raise TimeoutError(f"Services on instance {instance_id} not healthy within {timeout}s")


def run_test(instance_id: int) -> None:
    """Execute the end-to-end test task and print results.

    Args:
        instance_id: The target instance ID.
    """
    payload = json.dumps({
        "query": "Write a Python function that sorts a list of numbers, then summarize what it does"
    })
    cmd = (
        f"curl -s --max-time 300 -X POST http://localhost:8002/task "
        f"-H 'Content-Type: application/json' -d '{payload}'"
    )
    result = vastai("execute", str(instance_id), f'"{cmd}"', raw=False)
    response = json.loads(result) if isinstance(result, str) else result
    print("\n=== Test Result ===")
    print(f"Steps executed: {len(response.get('steps', []))}")
    print(f"Final output:\n{response.get('final_output', '(no output)')}")


def main() -> None:
    """Orchestrate the full deployment lifecycle."""
    parser = argparse.ArgumentParser(description="Deploy LoRA stack to vast.ai")
    parser.add_argument("--keep", action="store_true", help="Skip teardown after test")
    parser.add_argument("--instance-id", type=int, help="Use existing running instance")
    parser.add_argument("--gpu-ram", type=int, default=24, help="Minimum VRAM in GB")
    parser.add_argument("--disk", type=int, default=60, help="Disk size in GB")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set in environment or .env file", file=sys.stderr)
        sys.exit(1)

    instance_id = args.instance_id
    start = time.time()

    try:
        if instance_id is None:
            print(f"\n[1/6] Searching for GPU offer (>={args.gpu_ram}GB VRAM, {args.disk}GB disk)...")
            offer_id = find_offer(args.gpu_ram, args.disk)

            print(f"\n[2/6] Renting instance...")
            instance_id = create_instance(offer_id, args.disk)

            print(f"\n[3/6] Waiting for instance to start...")
            wait_until_running(instance_id)
        else:
            print(f"\n[--] Using existing instance {instance_id}, skipping creation.")

        print(f"\n[4/6] Copying source code...")
        copy_code(instance_id)

        print(f"\n[5/6] Installing dependencies and starting services...")
        setup_server(instance_id, hf_token)
        start_services(instance_id)

        print(f"\n[6/6] Waiting for services to be healthy (this may take several minutes)...")
        wait_healthy(instance_id)

        print(f"\n[TEST] Running end-to-end task...")
        run_test(instance_id)

        elapsed = time.time() - start
        print(f"\nDone in {elapsed:.0f}s. Instance: {instance_id}")

    finally:
        if not args.keep and instance_id is not None:
            print(f"\n[TEARDOWN] Destroying instance {instance_id}...")
            vastai("destroy", "instance", str(instance_id), "-y", parse=False)
            print("Instance destroyed.")
        elif args.keep:
            print(f"\n[KEEP] Instance {instance_id} left running.")


if __name__ == "__main__":
    main()
