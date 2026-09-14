"""Export model weights and their provenance without optimizer/RNG state."""

import argparse
import hashlib
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.checkpoint), Path(args.output)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    selected = {name: payload[name] for name in (
        "format_version", "config", "config_digest", "seed", "model", "global_step", "elapsed_seconds")}
    selected["inference_only"] = True
    selected["training_checkpoint_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(selected, handle)
    print(f"{output}: {output.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
