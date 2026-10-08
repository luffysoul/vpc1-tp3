# -*- coding: utf-8 -*-
"""Arma TP3_VPC1.ipynb a partir de las celdas de cells_tp3.py y lo ejecuta.

Uso:
    python build_nb.py            arma y ejecuta
    python build_nb.py --solo     arma sin ejecutar
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nbformat as nbf

from cells_tp3 import CELDAS

SALIDA = "TP3_VPC1.ipynb"


def armar():
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if tipo == "markdown" else nbf.v4.new_code_cell(src)
        for tipo, src in CELDAS
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    with io.open(SALIDA, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    n_md = sum(1 for t, _ in CELDAS if t == "markdown")
    n_co = len(CELDAS) - n_md
    print(f"{SALIDA}: {len(CELDAS)} celdas ({n_md} markdown, {n_co} codigo)")


def ejecutar():
    from nbclient import NotebookClient

    with io.open(SALIDA, encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                             resources={"metadata": {"path": "."}})
    cliente.execute()
    with io.open(SALIDA, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    errores = [(i, o["ename"], o["evalue"])
               for i, c in enumerate(nb["cells"])
               for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errores:
        for i, en, ev in errores:
            print(f"  ERROR celda {i}: {en}: {ev}")
        raise SystemExit(1)
    print("ejecutado sin errores")


if __name__ == "__main__":
    armar()
    if "--solo" not in sys.argv:
        ejecutar()
