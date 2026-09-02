# Build web assets for the male fly CNS console from MaleCNS v1.0.

# Source data (CC-BY 4.0, male-cns.janelia.org, released 2026-06-08):
#   body-annotations-male-cns-v1.0-minconf-0.5.feather
#   body-neurotransmitters-male-cns-v1.0.feather
#   connectome-weights-male-cns-v1.0-minconf-0.5.feather
#   syn-partners-male-cns-v1.0-minconf-0.5.feather
# Downloaded from https://storage.googleapis.com/flyem-male-cns/v1.0/...

# Outputs (public/data/connectomes/male-cns/):
#   meta.json      provenance, vocabularies, simulation config
#   neurons.bin    per-neuron table for the traced universe
#   names.bin      per-neuron instance names, length prefixed
#   matrix.bin     edges above the display weight threshold
#   cloud.bin      quantized synapse point cloud
#   sim_eye.bin    spike raster, eye stimulus
#   sim_ear.bin    spike raster, ear stimulus
#   stats.json     every recomputed number the page quotes

# Every number in stats.json is computed here from the raw files. Nothing is
# copied from secondary sources.

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import scipy.sparse as sp

EM_VOLUME_VOXELS = (94088, 78317, 134576)  # from male-cns.janelia.org/download/
EM_VOXEL_NM = 8

# Neurotransmitter sign for the presynaptic neuron. Unknown is treated as
# excitatory: the large majority of predicted neurons are cholinergic.
NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- annotations

ANNOTATION_COLS = [
    "bodyId", "class", "subclass", "superclass", "type", "instance",
    "somaSide", "status", "statusLabel", "somaLocation",
]


def load_annotations(path):
    log(f"reading {path.name}")
    t = feather.read_table(path, columns=ANNOTATION_COLS, memory_map=True)
    df = t.to_pandas().sort_values("bodyId").reset_index(drop=True)
    neurons = df[df["status"] == "Traced"].reset_index(drop=True)
    stats = {
        "segments_total": int(len(df)),
        "status_counts": {str(k): int(v) for k, v in
                          df["status"].value_counts(dropna=False).items()},
        "status_label_counts": {str(k): int(v) for k, v in
                                df["statusLabel"].value_counts(dropna=False).items()},
        "class_counts_all": {str(k): int(v) for k, v in
                             df["class"].value_counts(dropna=False).items()},
        "superclass_counts_all": {str(k): int(v) for k, v in
                                  df["superclass"].value_counts(dropna=False).items()},
        "traced_neurons": int(len(neurons)),
    }
    return df, neurons, stats


def build_vocab(values):
    # 255 is reserved for "<none>" in the shipped uint8 arrays.
    vals = sorted({str(v) for v in values if v is not None and str(v) != ""
                   and str(v) != "nan" and len(str(v)) < 200})
    return {v: i for i, v in enumerate(vals)}


def string_idx(series, vocab):
    out = np.full(len(series), 255, dtype=np.uint8)
    for i, v in enumerate(series):
        s = str(v) if v is not None else "<none>"
        if s == "nan" or s == "":
            continue
        j = vocab.get(s)
        if j is not None:
            out[i] = j
    return out


def neuron_arrays(neurons, super_vocab, class_vocab, cell_voxels):
    n = len(neurons)
    body_ids = neurons["bodyId"].to_numpy(dtype=np.int64)
    super_idx = string_idx(neurons["superclass"], super_vocab)
    class_idx = string_idx(neurons["class"], class_vocab)
    soma = np.zeros((n, 3), dtype=np.uint16)
    has_soma = np.zeros(n, dtype=np.uint8)
    for i, loc in enumerate(neurons["somaLocation"]):
        if loc is not None and len(loc) == 3:
            x, y, z = int(loc[0]), int(loc[1]), int(loc[2])
            if x >= 0 and y >= 0 and z >= 0:
                soma[i] = (x // cell_voxels, y // cell_voxels, z // cell_voxels)
                has_soma[i] = 1
    return body_ids, super_idx, class_idx, soma, has_soma


# ------------------------------------------------------------------- weights

def scan_weights(path, body_ids, sim_threshold, display_threshold,
                 batch_rows=16_000_000):
    log(f"scanning {path.name} ({path.stat().st_size / 1e9:.2f} GB) in batches")
    n = len(body_ids)
    total_rows = 0
    total_weight = 0
    hist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    out_weight = np.zeros(n, dtype=np.float64)
    in_degree = np.zeros(n, dtype=np.int64)
    out_degree = np.zeros(n, dtype=np.int64)
    sim_pre, sim_post, sim_w = [], [], []
    disp_pre, disp_post, disp_w = [], [], []

    t = feather.read_table(path, memory_map=True)
    for batch in t.to_batches(max_chunksize=batch_rows):
        b = np.asarray(batch.column("body_pre"))
        p = np.asarray(batch.column("body_post"))
        w = np.asarray(batch.column("weight"))
        total_rows += len(w)
        total_weight += int(w.sum())
        for k in hist:
            hist[k] += int((w >= k).sum())
        bi = np.searchsorted(body_ids, b)
        pi = np.searchsorted(body_ids, p)
        valid = (bi < n) & (pi < n)
        valid &= body_ids[np.minimum(bi, n - 1)] == b
        valid &= body_ids[np.minimum(pi, n - 1)] == p
        bi, pi, w2 = bi[valid], pi[valid], w[valid]
        out_weight += np.bincount(bi, weights=w2, minlength=n)
        in_degree += np.bincount(pi, minlength=n)
        out_degree += np.bincount(bi, minlength=n)
        m = w2 >= sim_threshold
        sim_pre.append(bi[m].astype(np.uint32))
        sim_post.append(pi[m].astype(np.uint32))
        sim_w.append(w2[m].astype(np.float32))
        m2 = w2 >= display_threshold
        disp_pre.append(bi[m2].astype(np.uint32))
        disp_post.append(pi[m2].astype(np.uint32))
        disp_w.append(np.minimum(w2[m2], 65535).astype(np.uint16))
        if total_rows // 64_000_000 != (total_rows - len(w)) // 64_000_000:
            log(f"  {total_rows / 1e6:.0f}M rows")

    pack = lambda arrs, dt: (np.concatenate(arrs).astype(dt) if arrs
                             else np.zeros(0, dt))
    result = {
        "rows": total_rows,
        "total_weight": total_weight,
        "threshold_counts": {str(k): v for k, v in hist.items()},
        "out_weight": out_weight,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "sim": (pack(sim_pre, np.uint32), pack(sim_post, np.uint32),
                pack(sim_w, np.float32)),
        "display": (pack(disp_pre, np.uint32), pack(disp_post, np.uint32),
                    pack(disp_w, np.uint16)),
    }
    log(f"  rows={result['rows']:,} sum(weight)={total_weight:,}")
    return result


# -------------------------------------------------------------- syn-partners

def scan_syn_partners(path, cell_voxels, body_ids, batch_rows=12_000_000):
    log(f"scanning {path.name} ({path.stat().st_size / 1e9:.2f} GB) in batches")
    dims = tuple(int(math.ceil(v / cell_voxels)) for v in EM_VOLUME_VOXELS)
    log(f"  grid dims {dims} at {cell_voxels * EM_VOXEL_NM} nm cells")
    plane = dims[1] * dims[2]
    n = len(body_ids)
    total_rows = 0
    key_parts, reg_parts = [], []
    pos_sum = np.zeros((n, 3), dtype=np.float64)
    pos_cnt = np.zeros(n, dtype=np.int64)
    # dominant presynaptic neuropil per body: counts per (body, region)
    REG_MAX = 256
    reg_counts = np.zeros((n, REG_MAX), dtype=np.int32)

    t = feather.read_table(path, columns=["x_pre", "y_pre", "z_pre", "body_pre",
                                          "primary_post"],
                           memory_map=True)
    for batch in t.to_batches(max_chunksize=batch_rows):
        x = np.asarray(batch.column("x_pre"))
        y = np.asarray(batch.column("y_pre"))
        z = np.asarray(batch.column("z_pre"))
        b = np.asarray(batch.column("body_pre"))
        col = batch.column("primary_post")
        reg = dictionary_indices(col, REGION_VOCAB_HOLDER)

        total_rows += len(x)
        bi = np.searchsorted(body_ids, b)
        bvalid = (bi < n) & (body_ids[np.minimum(bi, n - 1)] == b)
        for axis, arr in ((0, x), (1, y), (2, z)):
            pos_sum[:, axis] += np.bincount(bi[bvalid], weights=arr[bvalid],
                                            minlength=n)
        np.add.at(pos_cnt, bi[bvalid], 1)
        # dominant region per body: presynaptic-site counts per neuropil;
        # region ids above the 256 cap collapse into the top bucket.
        # accumulate per-batch UNIQUE (body, region) pairs: the record
        # batches are small, so a full-width bincount per batch is slow
        reg_c = np.minimum(reg[bvalid], REG_MAX - 1)
        combo = bi[bvalid].astype(np.int64) * REG_MAX + reg_c
        uq, cnt = np.unique(combo, return_counts=True)
        reg_flat = reg_counts.reshape(-1)
        reg_flat[uq] += cnt

        gx = (x // cell_voxels).astype(np.int64)
        gy = (y // cell_voxels).astype(np.int64)
        gz = (z // cell_voxels).astype(np.int64)
        ok = ((gx >= 0) & (gy >= 0) & (gz >= 0) &
              (gx < dims[0]) & (gy < dims[1]) & (gz < dims[2]))
        gx, gy, gz, reg = gx[ok], gy[ok], gz[ok], reg[ok]
        key_parts.append(gx * plane + gy * dims[2] + gz)
        reg_parts.append(reg)
        if total_rows // 64_000_000 != (total_rows - len(x)) // 64_000_000:
            log(f"  {total_rows / 1e6:.0f}M rows")

    keys = np.concatenate(key_parts)
    regs = np.concatenate(reg_parts)
    del key_parts, reg_parts
    uniq, first = np.unique(keys, return_index=True)
    count = np.bincount(uniq, minlength=int(np.prod(dims))).astype(np.uint16)
    region_full = np.zeros(int(np.prod(dims)), dtype=np.int16)
    region_full[uniq] = regs[first].astype(np.int16)
    del keys, regs, uniq, first
    nz = np.nonzero(count)[0]
    region_nz = region_full[nz]
    del region_full
    # key = gx * plane + gy * dims[2] + gz, so x is the MOST significant axis.
    gx = nz // plane
    rem = nz % plane
    gy = rem // dims[2]
    gz = rem % dims[2]
    result = {
        "rows": total_rows,
        "unique_tbars": int(len(nz)),
        "cells": int(len(nz)),
        "dims": dims,
        "count": count[nz],
        "region": region_nz,
        "x": gx.astype(np.uint16),
        "y": gy.astype(np.uint16),
        "z": gz.astype(np.uint16),
        "region_vocab": REGION_VOCAB_HOLDER["vocab"],
        "pos_sum": pos_sum,
        "pos_cnt": pos_cnt,
        "reg_counts": reg_counts,
    }
    log(f"  contacts={total_rows:,} unique TBars={len(nz):,}")
    return result


REGION_VOCAB_HOLDER = {"vocab": {}}


def dictionary_indices(col, holder):
    # col: a DictionaryArray (batch column) of dictionary<values=string>.
    # holder: dict with a "vocab" name-to-index map shared across batches.
    # Returns int64 region indices aligned with the rows.
    vocab = holder["vocab"]
    if isinstance(col, pa.ChunkedArray):
        parts = []
        offset = 0
        for chunk in col.chunks:
            part = _dict_indices_one(chunk, vocab)
            parts.append(part)
            offset += len(part)
        return np.concatenate(parts)
    return _dict_indices_one(col, vocab)


def _dict_indices_one(chunk, vocab):
    values = [str(v) for v in chunk.dictionary.to_pylist()]
    local = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if v not in vocab:
            vocab[v] = len(vocab)
        local[i] = vocab[v]
    idx = np.asarray(chunk.indices).astype(np.int64)
    return local[idx]


# ----------------------------------------------------------------------- sim

def build_sim_graph(pre, post, w, n, signs, mode):
    # CSR with rows = POSTSYNAPTIC neuron, columns = PRESYNAPTIC neuron, so
    # that graph.dot(spikes) delivers each firing source to its targets.
    # Weights are normalised per neuron: mode "in" divides by each target's
    # total input weight, so a neuron spikes when a fraction of the weight
    # feeding it fires together. mode "out" divides by the source's output.
    log(f"building signed simulation graph (norm={mode})")
    keep = w > 0
    pre, post, w = pre[keep], post[keep], w[keep]
    w = w.astype(np.float64)
    out_sum = np.bincount(pre, weights=w, minlength=n)
    in_sum = np.bincount(post, weights=w, minlength=n)
    denom = in_sum[post] if mode == "in" else out_sum[pre]
    w_signed = w * signs[pre].astype(np.float64) / np.maximum(denom, 1.0)
    w_signed = w_signed.astype(np.float32)
    order = np.lexsort((pre, post))
    post_r, pre_c, w_signed = post[order], pre[order], w_signed[order]
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(post_r, minlength=n)).astype(np.int32)
    graph = sp.csr_matrix((w_signed, pre_c.astype(np.int32), indptr),
                          shape=(n, n))
    log(f"  edges in sim graph: {graph.nnz:,}")
    return graph


def run_sim(graph, n, seeds, steps, dt_ms, leak, threshold, refractory, gain,
            noise, tag):
    log(f"simulating {tag}: {steps} steps x {dt_ms} ms, {graph.nnz:,} edges, "
        f"thr={threshold}, leak={leak}, gain={gain}, noise={noise}")
    rng = np.random.default_rng(42)
    v = np.zeros(n, dtype=np.float32)
    refr = np.zeros(n, dtype=np.int32)
    fired_hist = []
    transmissions = 0.0
    ext = np.zeros(n, dtype=np.float32)
    ext[seeds] = 1.0
    nnz_per_row = np.diff(graph.indptr)

    for step in range(steps):
        drive = ext * (3.0 if step < 12 else 0.0)
        s = ((v > threshold) & (refr <= 0))
        fired = np.nonzero(s)[0]
        if len(fired):
            transmissions += float(nnz_per_row[fired].sum())
            v[fired] = 0.0
            refr[fired] = refractory
        fired_hist.append(fired.astype(np.uint32))
        inflow = graph.dot(s.astype(np.float32))
        v = leak * v + gain * inflow + drive
        if noise > 0 and step % 4 == 0:
            v += rng.normal(0, noise, n).astype(np.float32)
        refr = np.maximum(refr - 1, 0)
        if step % 200 == 0:
            log(f"  step {step}, fired {len(fired)}")

    steps_arr, idx_arr = [], []
    for i, f in enumerate(fired_hist):
        if len(f):
            steps_arr.append(np.full(len(f), i, dtype=np.uint16))
            idx_arr.append(f)
    if steps_arr:
        raster = np.stack([np.concatenate(steps_arr),
                           np.concatenate(idx_arr)], axis=1)
    else:
        raster = np.zeros((0, 2), dtype=np.uint32)
    total = transmissions * (1000.0 / (steps * dt_ms))
    log(f"  total spikes {len(raster):,}, "
        f"transmissions {transmissions / 1e9:.2f} G "
        f"({total / 1e9:.2f} G per simulated second)")
    return raster, transmissions


def first_spike_ms(raster, mask, dt_ms):
    if len(raster) == 0:
        return None
    hit = raster[mask[raster[:, 1]]]
    if len(hit) == 0:
        return None
    return float(hit[:, 0].min() * dt_ms)


# ------------------------------------------------------------------- writer

def write_bin(path, arrays):
    with open(path, "wb") as f:
        for a in arrays:
            f.write(np.ascontiguousarray(a).tobytes())
    log(f"  wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cell-nm", type=int, default=4096)
    ap.add_argument("--display-threshold", type=int, default=20)
    ap.add_argument("--sim-threshold", type=int, default=3)
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fire-threshold", type=float, default=0.15)
    ap.add_argument("--leak", type=float, default=0.85)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.005)
    ap.add_argument("--norm", choices=["in", "out"], default="in")
    ap.add_argument("--wiring-edges", type=int, default=400000)
    ap.add_argument("--skip-cloud", action="store_true")
    args = ap.parse_args()

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = {"source": "MaleCNS v1.0 (male-cns.janelia.org, CC-BY 4.0)"}

    ann_path = indir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    df, neurons, ann_stats = load_annotations(ann_path)
    stats.update(ann_stats)
    super_vocab = build_vocab(neurons["superclass"])
    class_vocab = build_vocab(neurons["class"])
    stats["superclass_vocab"] = sorted(super_vocab, key=super_vocab.get)
    stats["class_vocab"] = sorted(class_vocab, key=class_vocab.get)

    body_ids, super_idx, class_idx, soma, has_soma = neuron_arrays(
        neurons, super_vocab, class_vocab, max(1, args.cell_nm // EM_VOXEL_NM))
    n = len(body_ids)
    log(f"traced neuron universe: {n:,}")

    nt_path = indir / "body-neurotransmitters-male-cns-v1.0.feather"
    log(f"reading {nt_path.name}")
    nt = feather.read_table(nt_path, columns=["body", "consensus_nt", "predicted_nt"],
                            memory_map=True).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    nt_name = nt["consensus_nt"].where(nt["consensus_nt"].notna(),
                                       nt["predicted_nt"]).fillna("unknown")
    nt_of = nt_name.reindex(pd_index(body_ids)).fillna("unknown")
    stats["nt_counts"] = {str(k): int(v) for k, v in nt_of.value_counts().items()}
    sign = np.array([NT_SIGN.get(s, 1.0) for s in nt_of], dtype=np.float32)

    w_path = indir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    wres = scan_weights(w_path, body_ids, args.sim_threshold, args.display_threshold)
    stats["connectome"] = {
        "rows": wres["rows"],
        "total_weight": wres["total_weight"],
        "threshold_counts": wres["threshold_counts"],
        "display_threshold": args.display_threshold,
        "sim_threshold": args.sim_threshold,
    }

    sp_path = indir / "syn-partners-male-cns-v1.0-minconf-0.5.feather"
    cloud = {"cells": 0, "dims": (0, 0, 0), "region_vocab": {"<unspecified>": 0}}
    if not args.skip_cloud:
        cloud = scan_syn_partners(sp_path, max(1, args.cell_nm // EM_VOXEL_NM),
                                  body_ids)
        stats["synapses"] = {
            "contact_rows": cloud["rows"],
            "occupied_cells": cloud["cells"],
            "cell_nm": args.cell_nm,
        }

    graph = build_sim_graph(*wres["sim"], n, sign, args.norm)

    types = neurons["type"].fillna("").astype(str)
    classes = neurons["class"].fillna("").astype(str)
    eye_mask = (types.str.match(r"^R[1-8]") & (classes == "visual")).to_numpy()
    ear_mask = types.str.startswith("JO").to_numpy()
    log(f"eye seeds: {int(eye_mask.sum())}, ear seeds: {int(ear_mask.sum())}")
    stats["stimuli"] = {"eye_neurons": int(eye_mask.sum()),
                        "ear_neurons": int(ear_mask.sum())}

    if not args.skip_cloud and "pos_sum" in cloud:
        pmean = np.zeros((n, 3), dtype=np.int32)
        pc = cloud["pos_cnt"]
        nonzero = pc > 0
        pmean[nonzero] = (cloud["pos_sum"][nonzero] / pc[nonzero, None]).astype(np.int32)
        has_pos = nonzero & (has_soma == 0)
        soma[has_pos] = pmean[has_pos].astype(np.uint16)
        has_soma[has_pos] = 1
        log(f"soma fallback from mean synapse position: {int(has_pos.sum()):,} neurons")
        stats["soma_fallback_synapse_mean"] = int(has_pos.sum())

        # dominant presynaptic neuropil per neuron (u8; 255 = none)
        reg_counts = cloud["reg_counts"]
        neuron_region = np.argmax(reg_counts, axis=1).astype(np.uint8)
        neuron_region[np.max(reg_counts, axis=1) == 0] = 255
        n_with_region = int((neuron_region < 255).sum())
        log(f"neurons with a dominant region: {n_with_region:,}")
        stats["neurons_with_region"] = n_with_region
    else:
        neuron_region = np.full(n, 255, dtype=np.uint8)

    # per-region neuron counts and cloud bounding boxes for the region chips
    vocab_items = sorted(cloud.get("region_vocab", {}).items(), key=lambda kv: kv[1])
    region_stats = []
    region_bounds = {}
    for name, ridx_v in vocab_items:
        cnt = int((neuron_region == ridx_v).sum())
        if cnt:
            region_stats.append({"name": name, "neurons": cnt})
    if not args.skip_cloud:
        cx_, cy_, cz_, creg = cloud["x"], cloud["y"], cloud["z"], cloud["region"]
        for name, ridx_v in vocab_items:
            sel = creg == ridx_v
            if not sel.any():
                continue
            region_bounds[name] = [
                int(cx_[sel].min()), int(cy_[sel].min()), int(cz_[sel].min()),
                int(cx_[sel].max()), int(cy_[sel].max()), int(cz_[sel].max())]
    stats["region_stats"] = region_stats
    stats["region_bounds"] = region_bounds

    sc = neurons["superclass"]
    stage_defs = [
        ("photoreceptor", eye_mask),
        ("optic_lobe", sc.isin(["ol_intrinsic", "ol_sensory", "visual_projection",
                                "visual_centrifugal"]).to_numpy()),
        ("central_brain", sc.isin(["cb_intrinsic", "cb_sensory", "cb_motor",
                                   "cb_endocrine", "ascending_neuron",
                                   "descending_neuron"]).to_numpy()),
        ("descending", sc.isin(["descending_neuron"]).to_numpy()),
        ("vnc", sc.isin(["vnc_intrinsic", "vnc_sensory", "vnc_motor",
                         "vnc_efferent", "vnc_endocrine"]).to_numpy()),
    ]

    sims = {}
    for tag, mask in [("eye", eye_mask), ("ear", ear_mask)]:
        if mask.sum() == 0:
            log(f"WARNING: no seed neurons for '{tag}', skipping sim")
            continue
        raster, transmissions = run_sim(
            graph, n, np.nonzero(mask)[0], args.steps, args.dt_ms,
            leak=args.leak, threshold=args.fire_threshold, refractory=8,
            gain=args.gain, noise=args.noise, tag=tag)
        stages = {}
        for name, smask in stage_defs:
            stages[name] = {"count": int(smask.sum()),
                            "first_spike_ms": first_spike_ms(raster, smask,
                                                             args.dt_ms)}
        sims[tag] = {
            "steps": args.steps, "dt_ms": args.dt_ms,
            "params": {"threshold": args.fire_threshold, "leak": args.leak,
                       "gain": args.gain, "noise": args.noise,
                       "norm": args.norm, "sim_threshold": args.sim_threshold},
            "transmissions": transmissions,
            "transmissions_per_second": transmissions * 1000.0 / (args.steps * args.dt_ms),
            "total_spikes": int(len(raster)),
            "stages": stages,
        }
        write_bin(outdir / f"sim_{tag}.bin",
                  [raster[:, 0].astype(np.uint16), raster[:, 1].astype(np.uint32)])
        stats.setdefault("simulation", {})[tag] = sims[tag]

    # neurons.bin v2, 24 bytes per neuron:
    # bodyId u32, inDeg u32, outDeg u32, superIdx u8, classIdx u8,
    # signCode u8 (1 exc, 2 inh), regionIdx u8 (255 none), stageIdx u8
    # (1 retina, 2 optic, 3 central, 4 descending, 5 VNC, 0 other),
    # hasPos u8, pos u16 x3
    sign_code = np.where(sign < 0, 2, 1).astype(np.uint8)
    stage_idx = np.zeros(n, dtype=np.uint8)
    for k, (name, smask) in enumerate(stage_defs):
        stage_idx[smask] = k + 1
    write_bin(outdir / "neurons.bin",
              [body_ids.astype(np.uint32), super_idx, class_idx, sign_code,
               neuron_region, stage_idx, has_soma, soma,
               np.minimum(wres["in_degree"], 4294967295).astype(np.uint32),
               np.minimum(wres["out_degree"], 4294967295).astype(np.uint32)])
    with open(outdir / "names.bin", "wb") as f:
        for s in neurons["instance"].fillna("").astype(str).tolist():
            b = s.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
    log(f"  wrote names.bin")

    dpre, dpost, dw = wres["display"]
    stats["connectome"]["display_edges_shipped"] = int(len(dpre))
    write_bin(outdir / "matrix.bin", [dpre, dpost, dw])

    # wiring.bin: the strongest edges of the SIMULATION graph (weight >= 3,
    # traced neurons, signed sources), for the neuron-to-neuron wiring view
    # and for lighting up signal travel during the spike replay.
    spre, spost, sw = wres["sim"]
    k = min(args.wiring_edges, len(spre))
    top = np.argpartition(sw, len(sw) - k)[len(sw) - k:]
    top = top[np.argsort(sw[top])[::-1]]
    write_bin(outdir / "wiring.bin",
              [spre[top].astype(np.uint32), spost[top].astype(np.uint32),
               np.minimum(sw[top], 65535).astype(np.uint16)])
    stats["wiring_edges_shipped"] = int(k)

    if not args.skip_cloud:
        write_bin(outdir / "cloud.bin", [cloud["x"], cloud["y"], cloud["z"],
                                         cloud["count"].astype(np.uint8),
                                         cloud["region"].astype(np.uint16)])

    stats["degree"] = {
        "max_out": int(wres["out_degree"].max()),
        "max_in": int(wres["in_degree"].max()),
    }
    meta = {
        "dataset": stats["source"],
        "release": "v1.0",
        "neuronFormat": 2,
        "regionBounds": stats["region_bounds"],
        "em_volume_voxels": EM_VOLUME_VOXELS,
        "em_voxel_nm": EM_VOXEL_NM,
        "cell_nm": args.cell_nm,
        "cloud_dims": list(cloud["dims"]),
        "region_vocab": sorted(cloud["region_vocab"], key=cloud["region_vocab"].get),
        "superclass_vocab": stats["superclass_vocab"] + ["<none>"],
        "class_vocab": stats["class_vocab"] + ["<none>"],
        "neurons": n,
        "simulation": sims,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (outdir / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log("done")


def pd_index(body_ids):
    import pandas as pd
    return pd.Index(body_ids)


if __name__ == "__main__":
    main()
