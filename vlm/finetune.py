"""LoRA fine-tune of Qwen3.5-4B on the drift-sense chat dataset."""
import subprocess
import sys

CMD = [
    sys.executable, "-m", "mlx_vlm.lora",
    "--model-path", "mlx-community/Qwen3.5-4B-MLX-4bit",
    "--dataset", "vlm/data",
    "--split", "train",
    "--epochs", "2",
    "--batch-size", "1",
    "--lora-rank", "8",
    "--learning-rate", "1e-4",
    "--output-path", "vlm/adapters",
]

if __name__ == "__main__":
    raise SystemExit(subprocess.call(CMD))
