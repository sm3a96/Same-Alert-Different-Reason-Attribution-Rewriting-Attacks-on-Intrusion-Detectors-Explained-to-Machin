"""Real detectors (Plan Phase 1, P1-3). Three families that dominate tabular NIDS:
tree (XGBoost), MLP, and attention (FT-Transformer). All implement the Detector
interface so they are drop-in for the toy detector everywhere.

FT-Transformer is implemented NATIVELY in torch (feature tokenizer + transformer
encoder + CLS head) rather than via the legacy `rtdl` package, which pins an old
torch and broke the CUDA stack. The two torch detectors are differentiable, exposing
`.module`, `.scaler`, and `to_input_tensor()` so Captum integrated gradients (Signal 1
smoothing, Signal 3) can attribute through them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from avert.detectors.base import Detector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class XGBoostDetector(Detector):
    """Tree family. Not differentiable; attribution via TreeSHAP."""

    name = "xgboost"

    def __init__(self, **params):
        self.params = dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            tree_method="hist", n_jobs=-1, eval_metric="logloss",
        )
        self.params.update(params)
        self._clf = None

    def fit(self, X, y):
        import xgboost as xgb

        # XGBoost demands labels 0..k-1 with no gaps. Any training subsample that misses a rare
        # class -- and CICIoT2023 has 34 classes, some of them tiny -- produces
        # "Invalid classes inferred from unique values of `y`" and kills the run. Encode to
        # contiguous codes here and decode on the way out, so callers keep the dataset's own
        # label space and nothing downstream has to know. A no-op when the labels are already
        # contiguous, which is the usual case, so this changes no existing result.
        y = np.asarray(y)
        self._classes = np.unique(y)
        self._code = {int(c): i for i, c in enumerate(self._classes)}
        self._clf = xgb.XGBClassifier(**self.params)
        self._clf.fit(X, np.array([self._code[int(v)] for v in y]))
        return self

    def predict(self, X):
        return self._classes[self._clf.predict(X)]

    def predict_proba(self, X):
        """Columns follow the DATASET's label space, so a class absent from training gets a
        zero column rather than shifting every other column left."""
        p = self._clf.predict_proba(X)
        if len(self._classes) and int(self._classes.max()) + 1 == len(self._classes):
            return p
        full = np.zeros((len(p), int(self._classes.max()) + 1), dtype=p.dtype)
        full[:, self._classes.astype(int)] = p
        return full

    @property
    def is_differentiable(self) -> bool:
        return False

    @property
    def booster(self):
        """Underlying model for shap.TreeExplainer.

        NB the booster speaks the ENCODED class space (0..k-1 over the classes present in
        training), while `predict` and `predict_proba` speak the dataset's label space. Anything
        that reads the booster directly -- TreeSHAP does -- must translate through
        `encoded_index`, or it will index a SHAP array by a dataset label and run off the end.
        """
        return self._clf

    def encoded_index(self, dataset_class: int) -> int | None:
        """Position of a dataset-space class in the booster's own output, or None if that class
        was absent from training and the booster therefore has no column for it."""
        return self._code.get(int(dataset_class))


# --------------------------------------------------------------------------- #
# Torch detectors (differentiable)                                            #
# --------------------------------------------------------------------------- #

class _TorchDetector(Detector):
    """Shared training/inference loop for the differentiable detectors."""

    def __init__(self, epochs: int = 30, lr: float = 1e-3, batch: int = 512, seed: int = 0):
        self.epochs, self.lr, self.batch, self.seed = epochs, lr, batch, seed
        self.scaler = StandardScaler()
        self.module: nn.Module | None = None
        self.n_classes: int | None = None

    def _build(self, d_in: int, n_classes: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        # Same gapped-label hazard as the tree detector, with a nastier failure. The head is
        # sized from the number of DISTINCT labels while the targets are raw dataset codes, so
        # a training subsample that misses a class hands CrossEntropyLoss a target beyond the
        # head and CUDA dies with "Assertion `t >= 0 && t < n_classes` failed" from inside a
        # kernel, several frames from the cause. Encode to contiguous codes and decode on the
        # way out. A no-op when the labels are already contiguous.
        y = np.asarray(y)
        self._classes = np.unique(y)
        code = {int(c): i for i, c in enumerate(self._classes)}
        self.n_classes = int(len(self._classes))
        self.module = self._build(X.shape[1], self.n_classes).to(DEVICE)
        opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(Xs),
            torch.from_numpy(np.array([code[int(v)] for v in y])).long()
        )
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True)
        self.module.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                lossf(self.module(xb), yb).backward()
                opt.step()
        self.module.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X, batch: int = 8192):
        """Columns follow the DATASET's label space; a class absent from training gets a zero
        column rather than shifting every other column left."""
        self.module.eval()
        Xs = self.scaler.transform(np.atleast_2d(X)).astype(np.float32)
        out = []
        for i in range(0, len(Xs), batch):           # batched to bound GPU memory
            xt = torch.from_numpy(Xs[i : i + batch]).to(DEVICE)
            out.append(torch.softmax(self.module(xt), dim=1).cpu().numpy())
        p = np.concatenate(out, axis=0)
        classes = getattr(self, "_classes", None)
        if classes is None or int(classes.max()) + 1 == len(classes):
            return p
        full = np.zeros((len(p), int(classes.max()) + 1), dtype=p.dtype)
        full[:, classes.astype(int)] = p
        return full

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def to_input_tensor(self, X) -> torch.Tensor:
        """Scaled input tensor on device — the space Captum attributes in."""
        Xs = self.scaler.transform(np.atleast_2d(X)).astype(np.float32)
        return torch.from_numpy(Xs).to(DEVICE)

    @property
    def is_differentiable(self) -> bool:
        return True


class _MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPDetector(_TorchDetector):
    name = "mlp"

    def __init__(self, hidden=(256, 128), **kw):
        super().__init__(**kw)
        self.hidden = hidden

    def _build(self, d_in, n_classes):
        return _MLP(d_in, n_classes, self.hidden)


class _EncoderBlock(nn.Module):
    """One transformer encoder block with explicit (matmul) multi-head attention, so we
    never touch torch's fused `_transformer_encoder_layer_fwd` kernel, which throws
    "CUDA error: invalid configuration argument" data-dependently on some GPUs. Sequence
    length here is tiny (1 + n_features), so manual attention costs nothing.
    """

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, "d_token must be divisible by n_heads"
        self.h, self.dh = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, L, d)
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B, h, L, dh)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        out = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, L, d)
        x = self.ln1(x + self.drop(self.proj(out)))
        return self.ln2(x + self.drop(self.ff(x)))


class _FTTransformer(nn.Module):
    """Numerical FT-Transformer (Gorishniy et al. 2021), native torch.

    Each numerical feature is tokenized to a d_token vector; a CLS token is prepended,
    manual-attention encoder blocks mix them, and the CLS output is classified.
    """

    def __init__(self, d_num: int, n_classes: int, d_token=64, n_blocks=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_num, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_num, d_token))
        self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        self.blocks = nn.ModuleList(_EncoderBlock(d_token, n_heads, dropout) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_classes))

    def forward(self, x):                                  # x: (B, d_num)
        tok = x.unsqueeze(-1) * self.weight + self.bias    # (B, d_num, d_token)
        seq = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        return self.head(seq[:, 0])                        # CLS head


class FTTransformerDetector(_TorchDetector):
    name = "ft_transformer"

    def __init__(self, d_token=64, n_blocks=3, n_heads=8, **kw):
        super().__init__(**kw)
        self.d_token, self.n_blocks, self.n_heads = d_token, n_blocks, n_heads

    def _build(self, d_in, n_classes):
        return _FTTransformer(d_in, n_classes, self.d_token, self.n_blocks, self.n_heads)
