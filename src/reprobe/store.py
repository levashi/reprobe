import json
import os
import torch
import h5py
from typing import Literal


class ActivationStore:
    """
    Persistent store for activations and labels, backed by a single HDF5 file.

    Prefill activations are pre-allocated (N, num_layers, hidden_dim).
    Token activations are stored with variable length per prompt via a resizable
    dataset + an index (start, end) per prompt for O(1) access.

    Survives crashes — reopen with resume=True to continue writing.

    Usage:
        store = ActivationStore("outputs/acts/store.h5", N=1000, mode="all")

        # in your collection loop:
        flushed = interceptor.flush_batch()
        store.append(
            acts=flushed,
            labels={"prefill": prefill_labels, "token": token_labels_list}
        )

        # pass to trainer:
        trainer = ProbesTrainer(model_id, hidden_dim, store=store)
        trainer.train_probes(mode="all", ...)
    """

    _TOKEN_CHUNK = 10_000  # token dataset grows by this many rows when full

    def __init__(
        self,
        path: str,
        N: int,
        mode: Literal["prefill", "token", "all"],
        resume: bool = False,
    ):
        """
        Args:
            path:   Path to the .h5 file. Created if it doesn't exist.
            N:      Number of prompts. Prefill uses this directly.
                    Token mode stores one variable-length entry per prompt.
            mode:   Which activation modes to store.
            resume: If True, reopen an existing file and continue writing
                    from the last cursor position. If False and the file
                    exists, it is overwritten.
        """
        self.path = path
        self.N = N
        self.mode = mode
        self._modes = ["prefill", "token"] if mode == "all" else [mode]
        self.cursors: dict[str, int] = {m: 0 for m in self._modes}

        self.num_layers: int | None = None
        self.hidden_dim: int | None = None
        self._initialized = False

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if resume and os.path.exists(path):
            self._resume()
        elif not resume and os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        acts: "dict[str, torch.Tensor | list[torch.Tensor] | None]",
        labels: "dict[str, torch.Tensor | list[torch.Tensor] | None]",
    ):
        """
        Write one batch of activations and labels.

        Args:
            acts:
                "prefill": Tensor[batch, num_layers, hidden_dim] | None
                "token":   list[Tensor[K_i, num_layers, hidden_dim]] | None
                           One tensor per prompt, K_i tokens each.
                           This is exactly what interceptor.flush_batch() returns.
            labels:
                "prefill": Tensor[batch] | None
                "token":   list[Tensor[K_i]] | None
                           One label tensor per prompt, matching token acts.
        """
        if not self._initialized:
            first = next(
                (t[0] if isinstance(t, list) else t)
                for t in acts.values() if t is not None
            )
            self._initialize(num_layers=first.shape[-2], hidden_dim=first.shape[-1])

        with h5py.File(self.path, "a") as f:
            for m in self._modes:
                a = acts.get(m)
                l = labels.get(m)
                if a is None or l is None:
                    continue
                if m == "prefill":
                    self._append_prefill(f, a, l)
                else:
                    self._append_token(f, a, l)

    def get_prefill(self) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Return all prefill activations and labels as tensors.
            acts:   Tensor[N, num_layers, hidden_dim]
            labels: Tensor[N]
        Loads everything into RAM — only call this from ProbesTrainer.
        """
        self._check_mode("prefill")
        n = self.cursors["prefill"]
        with h5py.File(self.path, "r") as f:
            acts = torch.from_numpy(f["prefill/acts"][:n].copy())
            labels = torch.from_numpy(f["prefill/labels"][:n].copy())
        return acts, labels

    def get_token_prompt(self, prompt_idx: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Return activations and labels for a single prompt.
            acts:   Tensor[K_i, num_layers, hidden_dim]
            labels: Tensor[K_i]
        """
        self._check_mode("token")
        with h5py.File(self.path, "r") as f:
            start, end = f["token/index"][prompt_idx]
            acts = torch.from_numpy(f["token/acts"][start:end].copy())
            labels = torch.from_numpy(f["token/labels"][start:end].copy())
        return acts, labels

    def iter_token_prompts(self):
        """
        Iterate over all token prompts, yielding (acts, labels) one prompt at a time.
        Never loads more than one prompt into RAM — use this in ProbesTrainer.
            acts:   Tensor[K_i, num_layers, hidden_dim]
            labels: Tensor[K_i]
        """
        self._check_mode("token")
        n = self.cursors["token"]
        with h5py.File(self.path, "r") as f:
            for i in range(n):
                start, end = f["token/index"][i]
                acts = torch.from_numpy(f["token/acts"][start:end].copy())
                labels = torch.from_numpy(f["token/labels"][start:end].copy())
                yield acts, labels

    @property
    def n_prefill(self) -> int:
        return self.cursors.get("prefill", 0)

    @property
    def n_token_prompts(self) -> int:
        return self.cursors.get("token", 0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initialize(self, num_layers: int, hidden_dim: int):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        with h5py.File(self.path, "a") as f:
            f.attrs["N"] = self.N
            f.attrs["mode"] = self.mode
            f.attrs["num_layers"] = num_layers
            f.attrs["hidden_dim"] = hidden_dim
            f.attrs["cursors"] = json.dumps(self.cursors)

            if "prefill" in self._modes and "prefill" not in f:
                grp = f.create_group("prefill")
                grp.create_dataset(
                    "acts",
                    shape=(self.N, num_layers, hidden_dim),
                    dtype="float32",
                )
                grp.create_dataset(
                    "labels",
                    shape=(self.N,),
                    dtype="float32",
                )

            if "token" in self._modes and "token" not in f:
                grp = f.create_group("token")
                chunk = min(self._TOKEN_CHUNK, 1024)
                grp.create_dataset(
                    "acts",
                    shape=(0, num_layers, hidden_dim),
                    maxshape=(None, num_layers, hidden_dim),
                    dtype="float32",
                    chunks=(chunk, num_layers, hidden_dim),
                )
                grp.create_dataset(
                    "labels",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float32",
                    chunks=(chunk,),
                )
                # (start, end) row index per prompt → O(1) access
                grp.create_dataset(
                    "index",
                    shape=(self.N, 2),
                    dtype="int64",
                )

        self._initialized = True

    def _append_prefill(self, f: "h5py.File", acts: "torch.Tensor", labels: "torch.Tensor"):
        a_np = acts.float().cpu().numpy()    # [batch, num_layers, hidden_dim]
        l_np = labels.float().cpu().numpy()  # [batch]
        batch = a_np.shape[0]
        cur = self.cursors["prefill"]

        if cur + batch > self.N:
            raise RuntimeError(
                f"ActivationStore prefill overflow: "
                f"tried to write {cur + batch} samples but N={self.N}."
            )

        f["prefill/acts"][cur : cur + batch] = a_np
        f["prefill/labels"][cur : cur + batch] = l_np
        self.cursors["prefill"] += batch
        self._save_cursors(f)

    def _append_token(
        self,
        f: "h5py.File",
        acts: "list[torch.Tensor]",
        labels: "list[torch.Tensor]",
    ):
        """
        acts:   list[Tensor[K_i, num_layers, hidden_dim]]  — one per prompt
        labels: list[Tensor[K_i]]                          — one per prompt
        """
        cur_prompt = self.cursors["token"]

        if cur_prompt + len(acts) > self.N:
            raise RuntimeError(
                f"ActivationStore token overflow: "
                f"tried to write prompt {cur_prompt + len(acts)} but N={self.N}."
            )

        token_ds = f["token/acts"]
        label_ds = f["token/labels"]
        index_ds = f["token/index"]

        for prompt_acts, prompt_labels in zip(acts, labels):
            a_np = prompt_acts.float().cpu().numpy()    # [K_i, num_layers, hidden_dim]
            l_np = prompt_labels.float().cpu().numpy()  # [K_i]
            K = a_np.shape[0]

            start = token_ds.shape[0]
            token_ds.resize(start + K, axis=0)
            label_ds.resize(start + K, axis=0)
            token_ds[start : start + K] = a_np
            label_ds[start : start + K] = l_np
            index_ds[cur_prompt] = (start, start + K)

            cur_prompt += 1

        self.cursors["token"] = cur_prompt
        self._save_cursors(f)

    def _save_cursors(self, f: "h5py.File"):
        f.attrs["cursors"] = json.dumps(self.cursors)

    def _resume(self):
        with h5py.File(self.path, "r") as f:
            self.N = int(f.attrs["N"])
            self.mode = str(f.attrs["mode"])
            self.num_layers = int(f.attrs["num_layers"])
            self.hidden_dim = int(f.attrs["hidden_dim"])
            self.cursors = json.loads(f.attrs["cursors"])
            self._modes = ["prefill", "token"] if self.mode == "all" else [self.mode]
        self._initialized = True