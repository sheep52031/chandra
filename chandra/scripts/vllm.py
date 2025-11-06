import os
import subprocess
import sys

from chandra.settings import settings


def main():
    cmd = [
        "sudo",
        "docker",
        "run",
        "--runtime",
        "nvidia",
        "--gpus",
        f"device={settings.VLLM_GPUS}",
        "-v",
        f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "--env",
        "VLLM_ATTENTION_BACKEND=TORCH_SDPA",
        "-p",
        "8000:8000",
        "--ipc=host",
        "vllm/vllm-openai:latest",
        "--model",
        settings.MODEL_CHECKPOINT,
        "--no-enforce-eager",
        "--max-num-seqs",
        "32",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max_num_batched_tokens",
        "65536",
        "--gpu-memory-utilization",
        ".9",
        "--served-model-name",
        settings.VLLM_MODEL_NAME,
    ]

    print(f"Starting vLLM server with command: {' '.join(cmd)}")

    try:
        # Use subprocess.run() which blocks and streams output automatically
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nvLLM server exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
