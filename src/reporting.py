"""LaTeX (booktabs) and Markdown table writers without extra dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s.replace("±", r"$\pm$").replace("->", r"$\rightarrow$")


def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, bold_best: dict | None = None) -> None:
    """``bold_best``: {column: 'max'|'min'} -> bold the best cell (compares the leading number)."""
    df = df.copy()
    if bold_best:
        for col, how in bold_best.items():
            if col not in df:
                continue
            vals = pd.to_numeric(df[col].astype(str).str.split(" ").str[0], errors="coerce")
            if vals.notna().any():
                best = vals.idxmax() if how == "max" else vals.idxmin()
                df.loc[best, col] = f"\\textbf{{{df.loc[best, col]}}}"
    cols = list(df.columns)
    caption = caption.replace("±", r"$\pm$")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = [str(v).replace("±", r"$\pm$") if str(v).startswith("\\textbf") else _esc(v) for v in row.values]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(map(str, row.values)) + " |")
    return "\n".join(out)
