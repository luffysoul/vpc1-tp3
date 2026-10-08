# -*- coding: utf-8 -*-
"""
Agente calificador del TP3 de Vision por Computadora I (CEIA 26Co2026).

Evalua la entrega como lo haria el docente de la materia, contra la rubrica del
enunciado: item 1 = 4 pts, item 2 = 4 pts, item 3 = 2 pts. Total 10.

El calificador hace DOS cosas que un docente haria y un checklist de palabras
clave no:

  PARTE A - corre el algoritmo entregado sobre las imagenes provistas y
            compara las detecciones contra la verdad de terreno. Los puntos de
            resultado se ganan detectando, no describiendo.
  PARTE B - abre el .ipynb y exige evidencia OBSERVABLE en las salidas
            guardadas de cada cosa que el enunciado pide documentar.

Ademas aplica las penalizaciones que el docente aplicaria sobre la entrega:
celdas con error, notebook que GitHub no renderiza, parametros hardcodeados por
imagen, ausencia de analisis de limitaciones.

Uso:  python calificador.py [notebook.ipynb]
"""
import io
import json
import os
import re
import sys

import nbformat as nbf

NOTEBOOK = sys.argv[1] if len(sys.argv) > 1 else "TP3_VPC1.ipynb"
BASE = os.path.dirname(os.path.abspath(__file__))
DIR_IMG = os.path.join(BASE, "data", "images")
RUTA_TPL = os.path.join(BASE, "data", "template", "pattern.png")
RUTA_GT = os.path.join(BASE, "data", "ground_truth.json")
IOU_ACIERTO = 0.5
LIMITE_MB = 1.6


# ===========================================================================
# Lectura del notebook
# ===========================================================================
class Notebook:
    def __init__(self, ruta):
        with io.open(ruta, encoding="utf-8") as f:
            self.nb = nbf.read(f, as_version=4)
        self.ruta = ruta
        self.celdas = self.nb["cells"]

    @property
    def codigo(self):
        return "\n".join(c["source"] for c in self.celdas if c["cell_type"] == "code")

    @property
    def markdown(self):
        return "\n".join(c["source"] for c in self.celdas if c["cell_type"] == "markdown")

    @property
    def salidas(self):
        partes = []
        for c in self.celdas:
            for out in c.get("outputs", []):
                if out.get("output_type") == "stream":
                    partes.append(out.get("text", ""))
                elif out.get("output_type") in ("execute_result", "display_data"):
                    d = out.get("data", {})
                    for k in ("text/plain", "text/html"):
                        if k in d:
                            v = d[k]
                            partes.append(v if isinstance(v, str) else "".join(v))
        return "\n".join(partes)

    @property
    def todo(self):
        return "\n".join([self.codigo, self.markdown, self.salidas])

    @property
    def errores(self):
        return [(i, o.get("ename", "?"), o.get("evalue", ""))
                for i, c in enumerate(self.celdas)
                for o in c.get("outputs", []) if o.get("output_type") == "error"]

    @property
    def n_figuras(self):
        n = 0
        for c in self.celdas:
            for o in c.get("outputs", []):
                d = o.get("data", {})
                if "image/png" in d or "image/jpeg" in d:
                    n += 1
        return n

    def celdas_ejecutadas(self):
        cc = [c for c in self.celdas if c["cell_type"] == "code" and c["source"].strip()]
        return sum(1 for c in cc if c.get("outputs")), len(cc)


# ===========================================================================
# PARTE A - verificacion funcional
# ===========================================================================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def verificar_funcionalmente():
    """Corre el detector entregado y compara contra la verdad de terreno."""
    res = {"ok": False, "motivo": "", "item1": {}, "multiples": {}, "determinista": None}
    try:
        sys.path.insert(0, BASE)
        import cv2 as cv
        import detector as D
    except Exception as e:                                   # pragma: no cover
        res["motivo"] = f"no se puede importar el detector: {e}"
        return res
    if not os.path.exists(RUTA_GT):
        res["motivo"] = "falta data/ground_truth.json"
        return res

    gt = json.load(io.open(RUTA_GT, encoding="utf-8"))
    prep = D.preparar_template(cv.imread(RUTA_TPL, cv.IMREAD_COLOR))

    for n in sorted(os.listdir(DIR_IMG)):
        img = cv.imread(os.path.join(DIR_IMG, n), cv.IMREAD_COLOR)
        H, W = img.shape[:2]

        una = D.detectar(img, prep, multiples=False)
        caja_ok = dentro = False
        v = 0.0
        if len(una) == 1:
            x, y, w, h = una[0]["caja"]
            v = iou((x, y, w, h), tuple(gt["item1"][n]))
            caja_ok = v >= IOU_ACIERTO
            # la caja tiene que estar mayormente dentro del cuadro y no ser absurda
            xi = max(0, min(x + w, W) - max(x, 0))
            yi = max(0, min(y + h, H) - max(y, 0))
            dentro = (w > 0 and h > 0 and xi * yi >= 0.5 * w * h)
        res["item1"][n] = {"n": len(una), "iou": v, "acierto": caja_ok and dentro}

        muchas = D.detectar(img, prep, multiples=True)
        esperado = gt["conteo_multiples"][n]
        solapan = any(iou(a["caja"], b["caja"]) > 0.5
                      for i, a in enumerate(muchas) for b in muchas[i + 1:])
        res["multiples"][n] = {"n": len(muchas), "esperado": esperado,
                               "ok": len(muchas) == esperado, "solapan": solapan}

    # determinismo: dos corridas iguales sobre la imagen mas dificil
    img = cv.imread(os.path.join(DIR_IMG, "coca_multi.png"), cv.IMREAD_COLOR)
    a = [d["caja"] for d in D.detectar(img, prep, multiples=True)]
    b = [d["caja"] for d in D.detectar(img, prep, multiples=True)]
    res["determinista"] = (a == b)
    res["ok"] = True
    return res


# ===========================================================================
# Rubrica
# ===========================================================================
class Rubrica:
    def __init__(self):
        self.items = []

    def parcial(self, seccion, criterio, puntos, fraccion, detalle=""):
        fraccion = max(0.0, min(1.0, float(fraccion)))
        self.items.append({"seccion": seccion, "criterio": criterio, "max": puntos,
                           "obtenido": puntos * fraccion,
                           "ok": fraccion >= 0.999, "detalle": detalle})

    def evaluar(self, seccion, criterio, puntos, condicion, detalle=""):
        self.parcial(seccion, criterio, puntos, 1.0 if condicion else 0.0, detalle)

    def total(self):
        return sum(i["obtenido"] for i in self.items), sum(i["max"] for i in self.items)


def calificar(ruta):
    nb = Notebook(ruta)
    fx = verificar_funcionalmente()
    r = Rubrica()
    sal, cod, todo = nb.salidas, nb.codigo, nb.todo

    # -----------------------------------------------------------------------
    S = "ITEM 1 - una deteccion en cada imagen, sin falsos positivos (4 pts)"
    if fx["ok"]:
        n_img = len(fx["item1"])
        exactamente_una = sum(1 for v in fx["item1"].values() if v["n"] == 1)
        aciertos = sum(1 for v in fx["item1"].values() if v["acierto"])
        r.parcial(S, "[funcional] Devuelve exactamente UNA deteccion por imagen", 1.0,
                  exactamente_una / n_img, f"{exactamente_una}/{n_img}")
        r.parcial(S, "[funcional] La deteccion cae sobre el logo (IoU>=0.5 vs verdad de terreno)",
                  2.0, aciertos / n_img,
                  f"{aciertos}/{n_img} — " + ", ".join(
                      f"{k}:{v['iou']:.2f}" for k, v in fx["item1"].items() if not v["acierto"]))
    else:
        r.parcial(S, "[funcional] Devuelve exactamente UNA deteccion por imagen", 1.0, 0,
                  fx["motivo"])
        r.parcial(S, "[funcional] La deteccion cae sobre el logo", 2.0, 0, fx["motivo"])

    r.evaluar(S, "Evidencia visual de las detecciones en el notebook", 1.0, nb.n_figuras >= 6,
              f"figuras con salida guardada: {nb.n_figuras}")

    # -----------------------------------------------------------------------
    S = "ITEM 2 - multiples detecciones en coca_multi.png (4 pts)"
    if fx["ok"]:
        m = fx["multiples"].get("coca_multi.png", {})
        r.parcial(S, "[funcional] Conteo correcto de instancias en coca_multi.png", 2.0,
                  1.0 if m.get("ok") else 0.0,
                  f"detectadas {m.get('n')} vs {m.get('esperado')} esperadas")
        r.evaluar(S, "[funcional] Las detecciones no se solapan entre si", 0.5,
                  not m.get("solapan", True), "sin NMS cada logo genera un racimo")
    else:
        r.parcial(S, "[funcional] Conteo correcto de instancias", 2.0, 0, fx["motivo"])
        r.parcial(S, "[funcional] Las detecciones no se solapan", 0.5, 0, fx["motivo"])

    hay_otsu = bool(re.search(r"otsu", todo, re.I))
    hay_cota = bool(re.search(r"cota de ruido|sqrt\(2|√\(2|2\s*\*\s*np\.log", todo, re.I))
    r.parcial(S, "El umbral de deteccion se determina automaticamente", 1.0,
              (0.5 if hay_otsu else 0) + (0.5 if hay_cota else 0),
              f"Otsu: {hay_otsu} · cota estadistica: {hay_cota}")

    hay_gt = bool(re.search(r"verdad de terreno|ground truth|contando a mano", todo, re.I))
    r.evaluar(S, "Validacion documentada contra la verdad de terreno", 0.5, hay_gt)

    # -----------------------------------------------------------------------
    S = "ITEM 3 - generalizacion a todas las imagenes (2 pts)"
    if fx["ok"]:
        n_img = len(fx["multiples"])
        bien = sum(1 for v in fx["multiples"].values() if v["ok"])
        r.parcial(S, "[funcional] El modo multiple da el conteo correcto en las 7", 1.5,
                  bien / n_img,
                  f"{bien}/{n_img} — " + ", ".join(
                      f"{k}:{v['n']}!={v['esperado']}" for k, v in fx["multiples"].items()
                      if not v["ok"]))
    else:
        r.parcial(S, "[funcional] El modo multiple da el conteo correcto en las 7", 1.5, 0,
                  fx["motivo"])
    r.evaluar(S, "Se tabula el conteo por imagen en el notebook", 0.5,
              bool(re.search(r"esperado", todo, re.I)))

    return nb, r, fx


# ===========================================================================
# Penalizaciones y chequeos de entrega
# ===========================================================================
def penalizaciones(nb, fx):
    p = []
    cod, todo = nb.codigo, nb.todo

    if nb.errores:
        p.append((2.0, f"{len(nb.errores)} celda(s) con error de ejecucion"))

    hechas, total = nb.celdas_ejecutadas()
    if total and (total - hechas) / total > 0.1:
        p.append((1.0, f"{total - hechas}/{total} celdas de codigo sin salida guardada"))

    # requisito literal del enunciado
    conf_dibujada = bool(re.search(r"putText", cod)) and bool(re.search(r"confianza", todo, re.I))
    if not conf_dibujada:
        p.append((1.0, "las cajas no muestran el nivel de confianza (lo pide el enunciado)"))

    # requisito transversal: nada ajustado por imagen
    patron = re.compile(r"['\"](?:COCA-COLA-LOGO\.jpg|coca_logo_[12]\.png|coca_multi\.png|"
                        r"coca_retro_[12]\.png|logo_1\.png)['\"]\s*:\s*[\d.]", re.I)
    if len(patron.findall(cod)) > 7:
        p.append((2.0, "parametros numericos indexados por nombre de imagen: "
                       "viola el requisito de automaticidad"))

    if not re.search(r"limitaci|qu[eé] no detecta|no detecta", todo, re.I):
        p.append((0.5, "no se discuten las limitaciones del metodo"))

    if fx.get("determinista") is False:
        p.append((1.0, "el detector no es determinista entre corridas"))

    mb = os.path.getsize(nb.ruta) / 1e6
    if mb > LIMITE_MB:
        p.append((1.0, f"el notebook pesa {mb:.2f} MB: GitHub no lo renderiza "
                       f"(limite ~{LIMITE_MB} MB) y la entrega es por repositorio"))

    carpeta = os.path.dirname(os.path.abspath(nb.ruta))
    if not os.path.exists(os.path.join(carpeta, "README.md")):
        p.append((0.5, "falta README.md con las instrucciones de uso"))
    if not os.path.exists(os.path.join(carpeta, "requirements.txt")):
        p.append((0.3, "falta requirements.txt con las versiones usadas"))

    return p


def informe(ruta):
    nb, r, fx = calificar(ruta)
    obtenido, maximo = r.total()
    penas = penalizaciones(nb, fx)

    print("=" * 88)
    print("AGENTE CALIFICADOR - Vision por Computadora I, TP3 (CEIA 26Co2026)")
    print(f"Notebook evaluado: {os.path.basename(ruta)}")
    print("=" * 88)

    if fx["ok"]:
        print("\nPARTE A - verificacion funcional (se corrio el algoritmo entregado)")
        print(f"  {'imagen':<24}{'item1 n':>9}{'IoU':>7}{'acierto':>9}"
              f"{'multiples':>11}{'esperado':>10}")
        print("  " + "-" * 70)
        for n in sorted(fx["item1"]):
            a, b = fx["item1"][n], fx["multiples"][n]
            print(f"  {n:<24}{a['n']:>9}{a['iou']:>7.2f}{str(a['acierto']):>9}"
                  f"{b['n']:>11}{b['esperado']:>10}")
        print(f"  determinista entre corridas: {fx['determinista']}")
    else:
        print(f"\nPARTE A - NO SE PUDO VERIFICAR: {fx['motivo']}")

    print("\nPARTE B - rubrica")
    for sec in dict.fromkeys(i["seccion"] for i in r.items):
        items = [i for i in r.items if i["seccion"] == sec]
        so = sum(i["obtenido"] for i in items)
        sm = sum(i["max"] for i in items)
        print(f"\n### {sec}  -  {so:.2f} / {sm:.2f}")
        for i in items:
            marca = "OK  " if i["ok"] else ("PARC" if i["obtenido"] > 0 else "FALL")
            print(f"  [{marca}] {i['obtenido']:.2f}/{i['max']:.2f}  {i['criterio']}")
            if i["detalle"] and not i["ok"]:
                print(f"           -> {i['detalle']}")

    print("\n" + "-" * 88)
    print(f"SUBTOTAL RUBRICA: {obtenido:.2f} / {maximo:.2f}")

    total_pena = sum(x for x, _ in penas)
    if penas:
        print("\nPENALIZACIONES:")
        for x, motivo in penas:
            print(f"  -{x:.2f}  {motivo}")
    else:
        print("\nPenalizaciones: ninguna.")

    hechas, total = nb.celdas_ejecutadas()
    mb = os.path.getsize(ruta) / 1e6
    print("\nCHEQUEOS DE ENTREGA:")
    print(f"  [{'OK' if not nb.errores else '--'}] todas las celdas ejecutan sin error")
    print(f"  [OK] celdas de codigo con salida: {hechas}/{total}")
    print(f"  [OK] figuras con salida guardada: {nb.n_figuras}")
    print(f"  [{'OK' if mb <= LIMITE_MB else '--'}] tamano del notebook: {mb:.2f} MB")

    nota = max(1.0, min(10.0, obtenido - total_pena))
    print("\n" + "=" * 88)
    print(f"NOTA FINAL: {nota:.2f} / 10")
    print("=" * 88)
    if nota >= 9:
        print("Veredicto: resuelve la consigna completa, valida cuantitativamente y documenta")
        print("sus limites. APROBADO PARA ENTREGA.")
    elif nota >= 7:
        print("Veredicto: correcto pero mejorable. Revisar los items PARC/FALL.")
    elif nota >= 4:
        print("Veredicto: aprobado al limite. Faltan partes sustantivas de la consigna.")
    else:
        print("Veredicto: insuficiente.")
    return nota


if __name__ == "__main__":
    raise SystemExit(0 if informe(NOTEBOOK) > 9 else 1)
