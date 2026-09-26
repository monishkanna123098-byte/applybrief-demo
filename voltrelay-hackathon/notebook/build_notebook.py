"""Assemble notebook/src/*.py (percent-format cells) into VoltRelay_Analysis.ipynb."""
from pathlib import Path

import jupytext

HERE = Path(__file__).parent
text = "\n\n".join(p.read_text() for p in sorted((HERE / "src").glob("*.py")))
nb = jupytext.reads(text, fmt="py:percent")
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
jupytext.write(nb, HERE / "VoltRelay_Analysis.ipynb")
print(f"Wrote {HERE / 'VoltRelay_Analysis.ipynb'} with {len(nb.cells)} cells")
