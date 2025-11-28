# proxy_tables.py
import os
import pandas as pd
import torch
from typing import Optional
_VALID_FILENAME_COLS = ["filename", "file", "image", "img", "name"]

# accepted extension swaps when we fail to find the exact name
_ALT_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

class ProxyTable:
    def __init__(self, table_path, node_names, zscore=True, alias_map=None):
        """
        table_path: path to CSV
        node_names: list of SCM node names (must match CSV cols, or use alias_map)
        alias_map: optional dict mapping SCM node -> csv_column_name
        """
        self.enabled = bool(table_path)
        self.node_names = list(node_names)
        self.alias_map = alias_map or {}
        if not self.enabled:
            return

        df = pd.read_csv(table_path)
        # find filename column
        fname_col = None
        for c in df.columns:
            if c.strip().lower() in _VALID_FILENAME_COLS:
                fname_col = c
                break
        if fname_col is None:
            raise ValueError(
                f"No filename column found in {table_path}. "
                f"Use one of: {', '.join(_VALID_FILENAME_COLS)}"
            )

        # canonicalize filename column
        df[fname_col] = (
            df[fname_col].astype(str)
            .str.strip()
            .str.lower()
            .map(lambda s: os.path.basename(s))
        )

        # map SCM node -> CSV column (with alias fallback)
        self.col_map = {}
        for n in self.node_names:
            csv_col = self.alias_map.get(n, n)
            # try exact, case-insensitive
            if csv_col not in df.columns:
                candidates = {c: c for c in df.columns if c.strip().lower() == csv_col.strip().lower()}
                if candidates:
                    csv_col = list(candidates.keys())[0]
                else:
                    # as a last resort, allow loose matching removing spaces/underscores
                    norm = lambda s: s.replace(" ", "").replace("_", "").lower()
                    norm_csv = norm(csv_col)
                    for c in df.columns:
                        if norm(c) == norm_csv:
                            csv_col = c
                            break
            if csv_col not in df.columns:
                print(f"[ProxyTable] WARNING: node '{n}' not found in CSV columns. Will return NaNs.")
                self.col_map[n] = None
            else:
                self.col_map[n] = csv_col

        # zscore if requested
        if zscore:
            for n, c in self.col_map.items():
                if c is None:
                    continue
                series = df[c].astype(float)
                mu = series.mean()
                std = series.std()
                if pd.isna(std) or std < 1e-8:
                    # avoid collapsing to NaN
                    df[c] = series - mu  # zero-mean only
                else:
                    df[c] = (series - mu) / (std + 1e-8)

        # store into torch tensors by filename for fast batch lookup
        self._by_name = {}
        for _, row in df.iterrows():
            name = row[fname_col]
            vals = {}
            for n, c in self.col_map.items():
                if c is None or pd.isna(row[c]):
                    vals[n] = float('nan')
                else:
                    vals[n] = float(row[c])
            self._by_name[name] = vals

    @staticmethod
    def _canon(fname: str) -> str:
        return os.path.basename(str(fname)).strip().lower()

    def _try_alt_exts(self, base: str) -> Optional[str]:
        """
        Try swapping extensions if exact name not found.
        base is lowercase basename (may include an ext).
        """
        root, ext = os.path.splitext(base)
        # try same name first (if caller didn't include ext)
        if base in self._by_name:
            return base
        # try every alt ext
        for e in _ALT_EXTS:
            cand = root + e
            if cand in self._by_name:
                return cand
        return None

    def get_batch(self, fnames, device="cpu"):
        """
        Returns dict {node_name: tensor(B,)} aligned to order of fnames.
        Missing entries become NaN.
        """
        B = len(fnames)
        out = {n: torch.full((B,), float('nan'), device=device) for n in self.node_names}
        for i, f in enumerate(fnames):
            key = self._canon(f)
            hit = key if key in self._by_name else self._try_alt_exts(key)
            if hit is None:
                continue
            row = self._by_name[hit]
            for n in self.node_names:
                v = row.get(n, float('nan'))
                out[n][i] = float(v)
        return out

    # Nice to have for debugging:
    def has(self, fname: str) -> bool:
        key = self._canon(fname)
        if key in self._by_name:
            return True
        return self._try_alt_exts(key) is not None
