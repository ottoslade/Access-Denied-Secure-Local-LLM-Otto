"""Build a tiny, random-weight llama GGUF for exercising the REAL llama-server.

The model produces gibberish, but llama.cpp loads it, serves it, reports
timings and uses real memory - enough to validate the launcher, gateway and
benchmark against the genuine engine without downloading weights.

Needs: pip install gguf numpy, and a vocab-only GGUF from the llama.cpp repo
(models/ggml-vocab-llama-spm.gguf).

    python scripts/dev/make_tiny_gguf.py --vocab path/to/ggml-vocab-llama-spm.gguf --out models/tiny-random-F32.gguf
    llama-quantize models/tiny-random-F32.gguf models/tiny-random-Q8_0.gguf Q8_0
"""

import argparse

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter


def copy_tokenizer(reader: GGUFReader, writer: GGUFWriter) -> int:
    n_vocab = 0
    for field in reader.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            sub = field.types[-1]
            if sub == GGUFValueType.STRING:
                val = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
            else:
                val = [field.parts[i].tolist()[0] for i in field.data]
            if field.name == "tokenizer.ggml.tokens":
                n_vocab = len(val)
            writer.add_array(field.name, val)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(field.name, bytes(field.parts[-1]).decode())
        else:
            v = field.parts[-1].tolist()[0]
            {GGUFValueType.UINT32: writer.add_uint32, GGUFValueType.INT32: writer.add_int32,
             GGUFValueType.BOOL: writer.add_bool, GGUFValueType.FLOAT32: writer.add_float32}.get(
                vtype, writer.add_uint32)(field.name, v)
    return n_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=768)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    w = GGUFWriter(args.out, "llama")
    w.add_name("tiny-random-llama (test only)")
    w.add_context_length(2048)
    w.add_embedding_length(args.embd)
    w.add_block_count(args.layers)
    w.add_feed_forward_length(args.ff)
    w.add_head_count(args.heads)
    w.add_head_count_kv(args.heads)
    w.add_rope_dimension_count(args.embd // args.heads)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_file_type(0)  # ALL_F32
    n_vocab = copy_tokenizer(GGUFReader(args.vocab), w)
    w.add_vocab_size(n_vocab)

    def t(name, *shape):
        w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32), raw_dtype=GGMLQuantizationType.F32)

    e, f = args.embd, args.ff
    t("token_embd.weight", n_vocab, e)
    w.add_tensor("output_norm.weight", np.ones(e, dtype=np.float32))
    t("output.weight", n_vocab, e)
    for i in range(args.layers):
        w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(e, dtype=np.float32))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(e, dtype=np.float32))
        for n in ("attn_q", "attn_k", "attn_v", "attn_output"):
            t(f"blk.{i}.{n}.weight", e, e)
        t(f"blk.{i}.ffn_gate.weight", f, e)
        t(f"blk.{i}.ffn_up.weight", f, e)
        t(f"blk.{i}.ffn_down.weight", e, f)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (vocab {n_vocab})")


if __name__ == "__main__":
    main()
