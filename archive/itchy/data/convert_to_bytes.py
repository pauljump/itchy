#!/usr/bin/env python3
"""Convert sp1024 tokenized shards to raw byte shards for Itchy.

Reads sp1024 shards, decodes tokens to text via SentencePiece, encodes to UTF-8 bytes,
and writes byte-level shards in the same binary format (header + uint16 values 0-255).
"""
import argparse
import struct
import numpy as np
import sentencepiece as spm
from pathlib import Path


def load_token_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens


def write_byte_shard(path: Path, byte_values: np.ndarray):
    """Write a shard in the same format: 256-int header + uint16 data."""
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(byte_values)  # num tokens (bytes in our case)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(byte_values.astype("<u2").tobytes())
    print(f"  Wrote {path}: {len(byte_values):,} bytes ({path.stat().st_size / 1e6:.1f}MB)")


def convert_shard(token_path: Path, sp: spm.SentencePieceProcessor, chunk_size: int = 100_000) -> np.ndarray:
    """Decode tokens to bytes in chunks to manage memory."""
    tokens = load_token_shard(token_path)
    print(f"  Loaded {token_path.name}: {len(tokens):,} tokens")

    all_bytes = []
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start:start + chunk_size].tolist()
        text = sp.decode(chunk)
        raw = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        all_bytes.append(raw)

    return np.concatenate(all_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=int, default=2)
    parser.add_argument("--sp-data", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--out-dir", default="./data/datasets/fineweb10B_bytes")
    args = parser.parse_args()

    sp_dir = Path(args.sp_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"Loaded tokenizer: {sp.vocab_size()} vocab")

    # Convert validation shard(s)
    val_files = sorted(sp_dir.glob("fineweb_val_*.bin"))
    for vf in val_files:
        print(f"\nConverting val shard: {vf.name}")
        byte_data = convert_shard(vf, sp)
        out_path = out_dir / vf.name
        write_byte_shard(out_path, byte_data)

    # Convert training shards
    train_files = sorted(sp_dir.glob("fineweb_train_*.bin"))[:args.train_shards]
    for tf in train_files:
        print(f"\nConverting train shard: {tf.name}")
        byte_data = convert_shard(tf, sp)
        out_path = out_dir / tf.name
        write_byte_shard(out_path, byte_data)

    print(f"\nDone! Byte shards in {out_dir}")
    # Show stats
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.bin"))
    print(f"Total disk: {total_bytes / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
