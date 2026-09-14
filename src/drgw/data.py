"""Public graph sources, auditable sparse preprocessing, and disjoint-node sampling.

Only topology is retained: directions, relation types, weights and self-loops are
removed. Full graphs remain CSR; dense tensors are made only for sampled graphs.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import weakref
import zipfile

import numpy as np
from scipy import sparse


def _snap(page, filename, nodes, edges, directed=False, **extra):
    return dict(source=f"https://snap.stanford.edu/data/{page}.html",
                url=f"https://snap.stanford.edu/data/{filename}",
                format="snap", source_nodes=nodes, source_edges=edges,
                directed=directed, status="available", **extra)


# Source pages checked on 2026-09-14. Published counts describe the source,
# not an assertion about the processed undirected simple graph.
DATASETS = {
    "Facebook": _snap("ego-Facebook", "facebook_combined.txt.gz", 4039, 88234),
    "Epinions": _snap("soc-Epinions1", "soc-Epinions1.txt.gz", 75879, 508837, True),
    "Pokec": _snap("soc-Pokec", "soc-pokec-relationships.txt.gz", 1632803, 30622564, True),
    "LiveJournal": _snap("soc-LiveJournal1", "soc-LiveJournal1.txt.gz", 4847571, 68993773, True),
    "Patents": dict(status="unresolved", source=None, url=None, format=None,
                    note="The manuscript's 23,133-node graph does not identify a unique public patent dataset."),
    "ogbn-arxiv": dict(source="https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
                       url="https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
                       format="ogb_csv", source_nodes=169343, source_edges=1166243,
                       directed=True, status="available"),
    "DBLP": _snap("com-DBLP", "com-dblp.ungraph.txt.gz", 317080, 1049866),
    "MIND": dict(status="available", source="https://github.com/HKUDS/DiffKG",
                 url="https://raw.githubusercontent.com/HKUDS/DiffKG/main/Datasets/mind/kg.txt",
                 format="triples", source_nodes=24733, source_edges=148568, directed=True,
                 note="Knowledge graph released with DiffKG; entity/triple counts match the manuscript."),
    "YAGO3-10": dict(source="https://github.com/TimDettmers/ConvE",
                     url="https://raw.githubusercontent.com/TimDettmers/ConvE/master/YAGO3-10.tar.gz",
                     format="yago", source_nodes=123182, source_edges=1089040,
                     directed=True, status="available",
                     note="Union of released train/valid/test triples; relation types discarded and parallel endpoints collapsed."),
    "ogbl-wikikg2": dict(source="https://ogb.stanford.edu/docs/linkprop/#ogbl-wikikg2",
                         url="https://snap.stanford.edu/ogb/data/linkproppred/wikikg-v2.zip",
                         format="ogb_csv", source_nodes=2500604, source_edges=17137181,
                         directed=True, status="available",
                         note="Uses raw graph training triples only; official link-prediction splits are not merged."),
    "Last-FM": dict(status="available", source="https://github.com/HKUDS/DiffKG",
                    url="https://raw.githubusercontent.com/HKUDS/DiffKG/main/Datasets/lastfm/kg.txt",
                    format="triples", source_nodes=58266, source_edges=464567, directed=True,
                    note="Knowledge graph released with DiffKG/KGAT, not SNAP LastFM Asia."),
    "Amazon": _snap("amazon0302", "amazon0302.txt.gz", 262111, 1234877, True,
                    note="March 2, 2003 co-purchasing graph; distinct from com-Amazon."),
    "ogbn-products": dict(source="https://ogb.stanford.edu/docs/nodeprop/#ogbn-products",
                          url="https://snap.stanford.edu/ogb/data/nodeproppred/products.zip",
                          format="ogb_csv", source_nodes=2449029, source_edges=61859140,
                          directed=False, status="available"),
    "web-Stanford": _snap("web-Stanford", "web-Stanford.txt.gz", 281903, 2312497, True),
    "BerkStan": _snap("web-BerkStan", "web-BerkStan.txt.gz", 685230, 7600595, True),
    "web-Google": _snap("web-Google", "web-Google.txt.gz", 875713, 5105039, True),
    "roadNet-TX": _snap("roadNet-TX", "roadNet-TX.txt.gz", 1379917, 1921660),
    "CA": _snap("roadNet-CA", "roadNet-CA.txt.gz", 1965206, 2766607),
}
REGISTRY = DATASETS
DATASETS["DBLP"]["url"] = "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
PAPER_COUNTS = dict(zip(DATASETS, [
    (4039, 88234), (75879, 405740), (1632803, 22301964), (4847571, 68993773),
    (23133, 93468), (169343, 1166243), (317080, 1049866), (24733, 148568),
    (123182, 1089040), (2500604, 17137170), (58266, 464567), (262111, 899792),
    (2449029, 61238448), (281903, 1992636), (685230, 6649470), (875713, 4322051),
    (1379917, 1921660), (1965206, 2766607)]))
PREPROCESS_VERSION = 1
SPLIT_SEED = 2026
SAMPLING_PROTOCOL = {
    "partition": "fixed-seed BFS traversal; consecutive 60%/20%/20% disjoint node pools",
    "partition_seed": SPLIT_SEED,
    "sample": "random-root BFS inside split pool; restart if exhausted; induced simple graph",
    "local_labels": "randomly permuted per sample; source IDs never used as model features",
    "overlap": "samples within one split may overlap; nodes across splits never overlap",
    "scope": "sampled-subgraph experiments, not whole-graph watermarking",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DRGW-research/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_edges(path, spec):
    if spec["format"] == "triples":
        return np.loadtxt(path, dtype=np.int64, usecols=(0, 2), ndmin=2), None
    if spec["format"] == "snap":
        with gzip.open(path, "rt") as handle:
            pairs = np.loadtxt(handle, dtype=np.int64, comments="#", usecols=(0, 1), ndmin=2)
        return pairs, None
    if spec["format"] == "yago":
        mapping, pairs = {}, []
        with tarfile.open(path, "r:gz") as archive:
            members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
            for filename in ("train.txt", "valid.txt", "test.txt"):
                with archive.extractfile(members[filename]) as handle:
                    for line in handle:
                        head, _, tail = line.decode("utf-8").strip().split("\t")
                        pairs.append((mapping.setdefault(head, len(mapping)),
                                      mapping.setdefault(tail, len(mapping))))
        return np.asarray(pairs, dtype=np.int64), len(mapping)
    if spec["format"] == "ogb_csv":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            edge_name = next(n for n in names if n.endswith("/raw/edge.csv.gz"))
            node_name = next(n for n in names if n.endswith("/raw/num-node-list.csv.gz"))
            with archive.open(edge_name) as zipped, gzip.open(zipped, "rt") as handle:
                pairs = np.loadtxt(handle, dtype=np.int64, delimiter=",", ndmin=2)
            with archive.open(node_name) as zipped, gzip.open(zipped, "rt") as handle:
                n = int(handle.read().strip())
        return pairs, n
    raise ValueError(f"Unsupported data format: {spec['format']}")


def load_dataset(name, data_dir="data", download=True):
    """Return ``(scipy.sparse.csr_matrix, provenance_dict)`` from a public source.

    Cached graph checksums are validated before use. A missing/unresolved source
    fails explicitly; synthetic data is never substituted for a named dataset.
    ``data_dir`` is the project's data root; downloads only enter its raw folder.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choices: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    if spec["status"] != "available":
        raise ValueError(f"{name}: unresolved source. {spec['note']}")
    root = Path(data_dir)
    processed = root / "processed" / name
    graph_path, meta_path = processed / "graph.npz", processed / "metadata.json"
    if graph_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("preprocess_version") != PREPROCESS_VERSION:
            raise ValueError(f"Stale preprocessing for {name}; remove {processed} and prepare again")
        if sha256_file(graph_path) != metadata["processed_sha256"]:
            raise ValueError(f"Checksum mismatch for {graph_path}")
        return sparse.load_npz(graph_path).tocsr(), metadata
    raw_path = root / "raw" / name / spec["url"].rsplit("/", 1)[-1]
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {raw_path}; enable download to prepare {name}")
        _download(spec["url"], raw_path)
    pairs, declared_n = _read_edges(raw_path, spec)
    raw_rows = len(pairs)
    loops = int(np.count_nonzero(pairs[:, 0] == pairs[:, 1]))
    if declared_n is None:
        node_ids, inverse = np.unique(pairs.reshape(-1), return_inverse=True)
        n = len(node_ids)
        pairs = inverse.reshape(-1, 2)
    else:
        n = declared_n
    graph = sparse.csr_matrix((np.ones(raw_rows, dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    graph = graph.maximum(graph.T)
    graph.setdiag(False)
    graph.eliminate_zeros()
    graph.sort_indices()
    processed.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(graph_path, graph, compressed=True)
    metadata = dict(name=name, source=spec, raw_file=str(raw_path.relative_to(root)),
                    raw_sha256=sha256_file(raw_path), raw_edge_rows=raw_rows,
                    raw_self_loop_rows=loops, nodes=n, edges=graph.nnz // 2,
                    paper_nodes=PAPER_COUNTS[name][0], paper_edges=PAPER_COUNTS[name][1],
                    preprocess_version=PREPROCESS_VERSION,
                    preprocessing=["reindex observed endpoints contiguously (OGB declared isolated nodes retained)",
                                   "discard edge types and attributes", "symmetrize by union", "remove self loops", "deduplicate edges"],
                    processed_sha256=sha256_file(graph_path),
                    prepared_at=datetime.now(timezone.utc).isoformat(),
                    sampling=SAMPLING_PROTOCOL)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return graph, metadata


_PARTITION_CACHE = {}


def split_node_pools(graph):
    """Deterministic topology-aware split; independent of all experiment seeds.

    BFS regions reduce the edge loss of random vertex splits on sparse roads.
    Cross-pool edges are unavailable in each induced graph. This split can cause
    distribution shift and must not be described as the manuscript's split.
    """
    graph = graph.tocsr(copy=False)
    key = id(graph)
    cached = _PARTITION_CACHE.get(key)
    if cached is not None and cached[0]() is graph:
        return cached[1]
    n = graph.shape[0]
    rng = np.random.default_rng(SPLIT_SEED)
    starts = rng.permutation(n)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    cursor = 0
    queue = deque()
    for start in starts:
        if visited[start]:
            continue
        visited[start] = True
        queue.append(int(start))
        while queue:
            node = queue.popleft()
            order[cursor] = node
            cursor += 1
            neighbors = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
            new = neighbors[~visited[neighbors]]
            visited[new] = True
            queue.extend(new.tolist())
    cut1, cut2 = int(n * 0.6), int(n * 0.8)
    pools = dict(train=order[:cut1], val=order[cut1:cut2], test=order[cut2:])
    _PARTITION_CACHE[key] = (weakref.ref(graph, lambda _: _PARTITION_CACHE.pop(key, None)), pools)
    return pools


def sample_subgraphs_with_metadata(graph, split, seed, num_graphs, num_nodes):
    """Return dense adjacencies and audit metadata containing source node IDs."""
    if num_graphs < 0 or num_nodes < 1:
        raise ValueError("num_graphs must be nonnegative and num_nodes positive")
    split = "val" if split in ("valid", "validation") else split
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    graph = graph.tocsr(copy=False)
    pool = split_node_pools(graph)[split]
    if len(pool) < num_nodes:
        raise ValueError(f"{split} pool has {len(pool)} nodes, fewer than requested {num_nodes}")
    allowed = np.zeros(graph.shape[0], dtype=bool)
    allowed[pool] = True
    rng = np.random.default_rng(seed)
    output = np.zeros((num_graphs, num_nodes, num_nodes), dtype=np.float32)
    ids = np.empty((num_graphs, num_nodes), dtype=np.int64)
    restarts = []
    for index in range(num_graphs):
        selected, queue = set(), deque()
        num_restarts = 0
        while len(selected) < num_nodes:
            if not queue:
                node = int(rng.choice(pool))
                while node in selected:
                    node = int(rng.choice(pool))
                queue.append(node)
                selected.add(node)
                num_restarts += 1
                if len(selected) == num_nodes:
                    break
            node = queue.popleft()
            neighbors = graph.indices[graph.indptr[node]:graph.indptr[node + 1]]
            neighbors = rng.permutation(neighbors[allowed[neighbors]])
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor not in selected:
                    selected.add(neighbor)
                    queue.append(neighbor)
                    if len(selected) == num_nodes:
                        break
        chosen = rng.permutation(sorted(selected))
        ids[index] = chosen
        output[index] = graph[chosen][:, chosen].toarray()
        restarts.append(num_restarts)
    metadata = dict(protocol=SAMPLING_PROTOCOL, split=split, seed=int(seed),
                    num_graphs=num_graphs, num_nodes=num_nodes, pool_nodes=len(pool),
                    source_node_ids=ids, bfs_components_started=restarts,
                    edge_counts=output.sum(axis=(1, 2)).astype(np.int64) // 2)
    return output, metadata


def sample_subgraphs(graph, split, seed, num_graphs, num_nodes):
    """Return float32 ``[G,N,N]`` induced graphs; every node is valid (no padding)."""
    return sample_subgraphs_with_metadata(graph, split, seed, num_graphs, num_nodes)[0]


def prepare_banks(config):
    """Create immutable, checksummed sampled-graph banks from a config mapping.

    Accept either the ``data`` section or a full config with a ``data`` key.
    Required keys: datasets, splits. Each split supplies num_nodes and num_graphs.
    Optional: data_dir (default data), seed (default 2026), download (default True).
    Return ``dict[dataset][split] = Path``. NPZ files contain uint8 ``adj`` and
    audit-only int64 ``node_ids``; models must not use node_ids as features.
    """
    if "data" in config:
        config = config["data"]
    root = Path(config.get("data_dir", "data"))
    datasets = list(config["datasets"])
    seed = int(config.get("seed", 2026))
    splits = config["splits"]
    if set(splits) - {"train", "val", "test"}:
        raise ValueError("Bank split names must be train, val and/or test")
    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate datasets in bank configuration")
    fingerprint_input = dict(datasets=datasets, seed=seed, splits=splits,
                             sampling=SAMPLING_PROTOCOL, preprocess_version=PREPROCESS_VERSION)
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()[:16]
    output = {}
    for dataset_index, name in enumerate(datasets):
        graph, provenance = load_dataset(name, root, download=config.get("download", True))
        dataset_dir = root / "banks" / fingerprint / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output[name] = {}
        for split, spec in splits.items():
            path = dataset_dir / f"{split}.npz"
            metadata_path = path.with_suffix(".json")
            split_seed = seed + dataset_index * 1000 + {"train": 0, "val": 1, "test": 2}[split]
            if path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata["source_graph_sha256"] != provenance["processed_sha256"]:
                    raise ValueError(f"Source graph changed for cached bank {path}; remove this bank and rebuild")
                if metadata["bank_sha256"] != sha256_file(path):
                    raise ValueError(f"Checksum mismatch for {path}")
            else:
                adj, metadata = sample_subgraphs_with_metadata(
                    graph, split, split_seed, int(spec["num_graphs"]), int(spec["num_nodes"]))
                node_ids = metadata.pop("source_node_ids")
                metadata["edge_counts"] = metadata["edge_counts"].tolist()
                np.savez_compressed(path, adj=adj.astype(np.uint8), node_ids=node_ids)
                pool = split_node_pools(graph)[split]
                metadata.update(dataset=name, bank_fingerprint=fingerprint,
                                source_graph_sha256=provenance["processed_sha256"],
                                bank_sha256=sha256_file(path),
                                split_pool_sha256=hashlib.sha256(pool.tobytes()).hexdigest())
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            output[name][split] = path
    return output
