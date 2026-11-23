import os, sys, json
# Redirect OGB/PyG cache to D: to protect the nearly-full C: drive
CACHE = "D:/2026/paper/PBB/post/data/ogb_cache"
os.makedirs(CACHE, exist_ok=True)
os.environ.setdefault("OGB_DIR", CACHE)
os.environ.setdefault("TORCH_GEOMETRIC_DIR", CACHE)

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects OGB's
# pre-processed files (PyG storage classes / numpy object arrays). OGB's data
# is a trusted source, so force weights_only=False for this trusted load.
import torch
_orig_load = torch.load
def _weights_off_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _weights_off_load

from ogb.nodeproppred import NodePropPredDataset
ds = NodePropPredDataset(name="ogbn-arxiv", root=CACHE)
g = ds.graph
spl = ds.get_idx_split()
y = ds.labels

num_nodes = g["num_nodes"]
edge_index = g["edge_index"]
feat = g["node_feat"]
out = {
    "num_nodes": int(num_nodes),
    "num_edges": int(edge_index.shape[1]),
    "feat_dim": int(feat.shape[1]),
    "num_classes": int(y.max()) + 1,
    "split_train": int(spl["train"].shape[0]),
    "split_valid": int(spl["valid"].shape[0]),
    "split_test": int(spl["test"].shape[0]),
    "feat_dtype": str(feat.dtype),
    "has_edge_feat": "edge_feat" in g,
    "cache_dir": CACHE,
}
print(json.dumps(out, indent=2))
