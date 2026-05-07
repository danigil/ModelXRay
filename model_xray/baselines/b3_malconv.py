"""B3: MalConv-lite raw-byte 1D-CNN.

Mirrors the architectural skeleton of MalConv (Raff et al., 2018) - learned byte
embedding, gated 1D convolution, global temporal max-pool, binary head - sized
small enough to fit on a single GPU at our sequence lengths (up to ~512 KB
windows on ResNet18; ~10 KB on SCZ-STL10).

Per Section 4.1 (Baseline) of the paper:
  - 8-d byte embedding
  - 2 parallel 1D convs of width 64, stride 16 (one ReLU feature, one sigmoid gate)
  - global temporal max-pool
  - 64-d FC head, ~1.5e5 params total
  - AdamW (lr 1e-3, wd 1e-4), BCE, early stopping on a held-out validation slice
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MalConvLite(nn.Module):
    def __init__(
        self,
        embed_dim: int = 8,
        n_filters: int = 64,
        kernel_size: int = 64,
        stride: int = 16,
        fc_hidden: int = 64,
    ):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.conv_main = nn.Conv1d(embed_dim, n_filters, kernel_size=kernel_size, stride=stride)
        self.conv_gate = nn.Conv1d(embed_dim, n_filters, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(n_filters, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        # byte_seq: (B, L) uint8 or long; cast on-device so host->device is 1 byte/elem.
        if byte_seq.dtype != torch.long:
            byte_seq = byte_seq.long()
        x = self.embed(byte_seq).transpose(1, 2)        # (B, E, L)
        feat = torch.relu(self.conv_main(x))
        gate = torch.sigmoid(self.conv_gate(x))
        h = (feat * gate).amax(dim=2)                    # (B, n_filters)
        h = torch.relu(self.fc1(h))
        return self.fc2(h).squeeze(-1)                   # (B,) logits


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 6
    val_frac: float = 0.15
    seed: int = 0
    device: str = "cuda"
    embed_dim: int = 8
    n_filters: int = 64
    kernel_size: int = 64
    stride: int = 16
    fc_hidden: int = 64


def _seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(X: np.ndarray, y: np.ndarray, device: str):
    Xt = torch.from_numpy(X).to(device, non_blocking=True)
    yt = torch.from_numpy(y).float().to(device, non_blocking=True)
    return Xt, yt


def _iter_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, generator: torch.Generator):
    n = X.shape[0]
    perm = torch.randperm(n, generator=generator, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], y[idx]


def _eval_gpu(model: MalConvLite, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            p = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.int64)
            preds.append(p)
            labels.append(yb.cpu().numpy().astype(np.int64))
    p = np.concatenate(preds)
    l = np.concatenate(labels)
    return float((p == l).mean()), p, l


def _per_class(pred, label):
    pos, neg = (label == 1), (label == 0)
    am = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    ab = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return ab, am


def train_and_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: TrainConfig,
    extra_tests: dict | None = None,
) -> dict:
    """Train MalConv-lite, evaluate on (X_test, y_test), optionally on extras.

    X_* are uint8 (N, L); y_* are 0/1. `extra_tests` is {name: (X, y)} of
    additional test sets scored with the SAME trained model.
    """
    assert X_train.dtype == np.uint8 and X_test.dtype == np.uint8
    if extra_tests:
        for n, (Xe, _) in extra_tests.items():
            assert Xe.dtype == np.uint8, f"extra_tests[{n!r}] X must be uint8"
    _seed_all(cfg.seed)

    rng = np.random.default_rng(cfg.seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    n_val_pos = max(1, int(len(pos_idx) * cfg.val_frac))
    n_val_neg = max(1, int(len(neg_idx) * cfg.val_frac))
    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    tr_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])

    device = cfg.device if torch.cuda.is_available() else "cpu"
    Xtr_d, ytr_d = _to_device(X_train[tr_idx], y_train[tr_idx], device)
    Xva_d, yva_d = _to_device(X_train[val_idx], y_train[val_idx], device)
    Xte_d, yte_d = _to_device(X_test, y_test, device)
    Xall_d, yall_d = _to_device(X_train, y_train, device)

    model = MalConvLite(
        embed_dim=cfg.embed_dim, n_filters=cfg.n_filters,
        kernel_size=cfg.kernel_size, stride=cfg.stride, fc_hidden=cfg.fc_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss()

    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed)

    best_val = -1.0
    best_state = None
    bad = 0
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum, n_seen = 0.0, 0
        for xb, yb in _iter_batches(Xtr_d, ytr_d, cfg.batch_size, shuffle=True, generator=g):
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss = loss_sum / max(n_seen, 1)
        val_acc, _, _ = _eval_gpu(model, Xva_d, yva_d, cfg.batch_size)
        history.append((epoch, train_loss, val_acc))
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, te_pred, te_label = _eval_gpu(model, Xte_d, yte_d, cfg.batch_size)
    train_acc, _, _ = _eval_gpu(model, Xall_d, yall_d, cfg.batch_size)
    acc_ben, acc_mal = _per_class(te_pred, te_label)

    extras = {}
    if extra_tests:
        for name, (Xe, ye) in extra_tests.items():
            Xe_d, ye_d = _to_device(Xe, ye, device)
            ext_acc, ext_pred, ext_label = _eval_gpu(model, Xe_d, ye_d, cfg.batch_size)
            ext_ben, ext_mal = _per_class(ext_pred, ext_label)
            extras[name] = {
                "test_acc": ext_acc,
                "acc_benign_test": ext_ben,
                "acc_malicious_test": ext_mal,
            }
            del Xe_d, ye_d

    del model, opt, Xtr_d, ytr_d, Xva_d, yva_d, Xte_d, yte_d, Xall_d, yall_d
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "best_val_acc": best_val,
        "epochs_run": history[-1][0] + 1 if history else 0,
        "acc_benign_test": acc_ben,
        "acc_malicious_test": acc_mal,
        "extras": extras,
    }
