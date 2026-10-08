# -*- coding: utf-8 -*-
"""
TP3 - Vision por Computadora I (CEIA, FIUBA)
Deteccion del logotipo de Coca-Cola a partir de un unico template.

Arquitectura: dos detectores complementarios y una regla de fusion automatica.

  A. SIFT + ratio test de Lowe + homografia por RANSAC.
     Invariante a escala, rotacion y perspectiva. Es el unico que tolera la
     deformacion cilindrica de una lata. Necesita que el logo sea lo bastante
     grande como para generar keypoints.

  B. Correlacion normalizada multiescala sobre niveles de gris.
     Soporta instancias chicas y repetidas, donde SIFT se queda sin keypoints.
     Asume el objeto fronto-paralelo y sin rotacion.

  Fusion: la confianza de A se mide por la cantidad de inliers de RANSAC. Si A
  es confiable, localiza el; si no, decide B.

Todos los parametros se derivan del par (template, imagen). No hay ningun valor
ajustado por imagen. Las constantes globales son adimensionales y cada una esta
justificada en su punto de uso.
"""
import numpy as np
import cv2 as cv

PISO_VAR = 0.01       # piso de varianza de la NCC, como fraccion de la energia del template
IOU_NMS = 0.30        # solape maximo entre dos detecciones distintas
RATIO_LOWE = 0.80     # ratio test, valor del paper original de Lowe (2004)
MIN_INLIERS = 8       # 2x el minimo algebraico de 4 que exige una homografia
PASO_ESCALA = 1.06    # paso multiplicativo del barrido grueso
PASO_FINO = 1.02      # paso del barrido guiado por SIFT


# ===========================================================================
# 1. Preparacion del template
# ===========================================================================
def color_de_fondo(gris):
    """Valor modal del marco: es el fondo sobre el que esta dibujado el logo."""
    borde = np.concatenate([gris[0], gris[-1], gris[:, 0], gris[:, -1]])
    return int(np.bincount(borde).argmax())


def mascara_de_tinta(bgr):
    """Separa el logo de su fondo por Otsu sobre la distancia al color de fondo."""
    gris = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    dif = cv.absdiff(gris, np.full_like(gris, color_de_fondo(gris)))
    _, m = cv.threshold(dif, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return m


def recortar_template(bgr):
    """Recorta el template a la caja de su contenido.

    El archivo provisto mide 400x175 (aspecto 2.29) pero el logo ocupa solo
    373x115 (aspecto 3.24): los margenes blancos deforman el aspecto un 41%.
    Sin este recorte la busqueda multiescala compara formas distintas.
    """
    ys, xs = np.where(mascara_de_tinta(bgr) > 0)
    return bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def semiancho_de_trazo(bgr):
    """Semiancho tipico del trazo via transformada de distancia sobre la tinta.

    Fija hasta donde se puede reducir el template antes de que el trazo se
    pierda: es el limite de muestreo que impone el propio logo, no un numero
    elegido a mano.
    """
    tinta = mascara_de_tinta(bgr)
    dt = cv.distanceTransform(tinta, cv.DIST_L2, 5)
    return float(np.percentile(dt[tinta > 0], 90))


# ===========================================================================
# 2. Correlacion normalizada regularizada e invariante a polaridad
# ===========================================================================
def a_gris(bgr):
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32)


def mapa_de_bordes(bgr, sigma=1.0):
    """Modulo del gradiente. Se usa para comparar representaciones en el analisis."""
    g = cv.GaussianBlur(a_gris(bgr), (0, 0), sigma)
    gx = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)
    return cv.magnitude(gx, gy)


def ncc_regularizada(F, T, piso=PISO_VAR):
    """NCC con piso de varianza atado a la energia del template.

    cv.TM_CCOEFF_NORMED divide por la desviacion de la ventana: sobre una zona
    plana el cociente es 0/0 y OpenCV devuelve 1.0, con lo que el fondo blanco
    gana siempre. Sumando piso*ss_T al denominador, una ventana sin estructura
    puntua ~0 en lugar de 1.
    """
    T = T - T.mean()
    ss_T = float((T * T).sum())
    if ss_T <= 0:
        return None
    n = T.size
    unos = np.ones_like(T)
    cruz = cv.matchTemplate(F, T, cv.TM_CCORR)
    s1 = cv.matchTemplate(F, unos, cv.TM_CCORR)
    s2 = cv.matchTemplate(F * F, unos, cv.TM_CCORR)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = np.sqrt((var + piso * ss_T) * ss_T)
    return cruz / np.maximum(den, 1e-12)


def psr(R, loc, guarda):
    """Peak-to-sidelobe ratio: cuantas sigmas sobresale el pico de su fondo.

    El maximo crudo de la NCC no es comparable entre escalas (un template chico
    puntua mas alto por azar y sobre un mapa mas grande). El PSR si lo es,
    porque normaliza cada respuesta contra su propia distribucion.
    """
    x, y = loc
    m = np.ones(R.shape, bool)
    m[max(0, y - guarda):y + guarda + 1, max(0, x - guarda):x + guarda + 1] = False
    lobulo = R[m]
    if lobulo.size < 50:
        return np.nan
    sd = float(lobulo.std())
    if sd < 1e-9:
        return np.nan
    return (float(R[y, x]) - float(lobulo.mean())) / sd


# ===========================================================================
# 3. Rango de escalas
# ===========================================================================
def rango_de_escalas(forma_img, forma_tpl, semi_trazo):
    """Cota superior por geometria, inferior por legibilidad del trazo."""
    H, W = forma_img[:2]
    th, tw = forma_tpl[:2]
    return 1.0 / semi_trazo, min(W / tw, H / th)


def grilla_de_escalas(s_min, s_max, paso=PASO_ESCALA):
    """Progresion geometrica: la escala es multiplicativa, el muestreo tambien."""
    if s_min >= s_max:
        return np.array([s_max])
    n = max(2, int(np.ceil(np.log(s_max / s_min) / np.log(paso))) + 1)
    return np.geomspace(s_min, s_max, n)


# ===========================================================================
# 4. Supresion de no maximos
# ===========================================================================
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(detecciones, umbral_iou=IOU_NMS):
    quedan = []
    for d in sorted(detecciones, key=lambda d: -d["score"]):
        if all(iou(d["caja"], k["caja"]) <= umbral_iou for k in quedan):
            quedan.append(d)
    return quedan


# ===========================================================================
# 5. Barrido multiescala  (detector B)
# ===========================================================================
def barrido_multiescala(F, T0, escalas):
    """Mejor pico por escala, usando |NCC|.

    Invertir el contraste cambia el signo de la correlacion, no su modulo. Por
    eso |NCC| es invariante a polaridad y el signo del pico revela cual de las
    dos se dio, sin tener que correr el barrido dos veces.
    """
    H, W = F.shape
    th0, tw0 = T0.shape
    salida = []
    for s in escalas:
        tw, th = int(round(tw0 * s)), int(round(th0 * s))
        if tw < 8 or th < 8 or tw > W or th > H:
            continue
        interp = cv.INTER_AREA if s < 1 else cv.INTER_CUBIC
        T = cv.resize(T0, (tw, th), interpolation=interp)
        R = ncc_regularizada(F, T)
        if R is None or not np.isfinite(R).all():
            continue
        A = np.abs(R)
        _, mx, _, loc = cv.minMaxLoc(A)
        z = psr(A, loc, guarda=max(tw, th) // 2)
        if np.isnan(z):
            continue
        salida.append({"escala": float(s), "psr": float(z), "ncc": float(mx),
                       "polaridad": "directa" if R[loc[1], loc[0]] > 0 else "invertida",
                       "caja": (int(loc[0]), int(loc[1]), tw, th), "R": A})
    return salida


# ===========================================================================
# 6. SIFT + homografia  (detector A)
# ===========================================================================
def correspondencias(tpl_gris, img_gris, ratio=RATIO_LOWE):
    sift = cv.SIFT_create()
    k1, d1 = sift.detectAndCompute(tpl_gris, None)
    k2, d2 = sift.detectAndCompute(img_gris, None)
    if d1 is None or d2 is None or len(k2) < 2:
        return [], k1, k2
    bf = cv.BFMatcher()
    buenas = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < ratio * n.distance]
    return buenas, k1, k2


def caja_desde_homografia(H, forma_tpl):
    """Proyecta las esquinas del template: da el cuadrilatero orientado."""
    th, tw = forma_tpl[:2]
    esq = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    return cv.perspectiveTransform(esq, H).reshape(-1, 2)


def cuadrilatero_valido(q, forma_tpl, tol_aspecto=2.5):
    """Rechaza homografias degeneradas sin mirar la cantidad de inliers.

    RANSAC siempre devuelve algo con 4 correspondencias. Un ajuste espurio
    colapsa el cuadrilatero o lo entrega con un aspecto imposible, y eso se
    detecta geometricamente.
    """
    if q is None or not np.isfinite(q).all():
        return False
    qf = q.astype(np.float32)
    if not cv.isContourConvex(qf):
        return False
    if abs(cv.contourArea(qf)) < 1.0:
        return False
    th, tw = forma_tpl[:2]
    lados = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    ancho = (lados[0] + lados[2]) / 2.0
    alto = (lados[1] + lados[3]) / 2.0
    if alto < 1e-6:
        return False
    razon = (ancho / alto) / (tw / float(th))
    return 1.0 / tol_aspecto <= razon <= tol_aspecto


def escala_votada(buenas, k1, k2, mascara):
    """Mediana de la razon de escalas de los keypoints inliers.

    Es el voto de escala del esquema de Hough generalizado de Lowe: cada
    correspondencia aporta su estimacion y la mediana la vuelve robusta.
    """
    sel = mascara.ravel().astype(bool)
    razones = [k2[m.trainIdx].size / k1[m.queryIdx].size
               for m, ok in zip(buenas, sel) if ok]
    return float(np.median(razones)) if razones else None


def pose_sift(tpl_bgr, img_bgr, excluir=None):
    """Mejor pose del template en la imagen, probando las dos polaridades.

    SIFT describe orientaciones de gradiente, que rotan 180 grados al invertir
    el contraste; por eso la polaridad hay que elegirla, y el criterio es la
    cantidad de inliers de RANSAC.

    `excluir` es el conjunto de indices de keypoints de la imagen ya usados por
    detecciones previas: habilita la busqueda de instancias sucesivas.
    """
    t = cv.cvtColor(tpl_bgr, cv.COLOR_BGR2GRAY)
    g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mejor = {"inliers": 0, "polaridad": None, "H": None, "escala": None,
             "n_buenas": 0, "usados": set()}
    for nombre, tt in (("directa", t), ("invertida", 255 - t)):
        buenas, k1, k2 = correspondencias(tt, g)
        if excluir:
            buenas = [m for m in buenas if m.trainIdx not in excluir]
        if len(buenas) < 4:
            continue
        src = np.float32([k1[m.queryIdx].pt for m in buenas]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in buenas]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
        if H is None or mask is None:
            continue
        n_in = int(mask.sum())
        if n_in > mejor["inliers"]:
            sel = mask.ravel().astype(bool)
            mejor = {"inliers": n_in, "polaridad": nombre, "H": H,
                     "escala": escala_votada(buenas, k1, k2, mask),
                     "n_buenas": len(buenas),
                     "usados": {m.trainIdx for m, ok in zip(buenas, sel) if ok}}
    return mejor


def poses_sift_iterativas(tpl_bgr, img_bgr, max_inst=40):
    """Multiples instancias: se detecta una, se retiran sus inliers y se repite.

    Una homografia modela una sola instancia; con varias copias del objeto
    RANSAC se queda con la que mas votos junta. Al retirar sus keypoints, la
    pasada siguiente encuentra la copia siguiente.
    """
    usados, salida = set(), []
    for _ in range(max_inst):
        r = pose_sift(tpl_bgr, img_bgr, excluir=usados)
        if r["inliers"] < MIN_INLIERS or r["H"] is None:
            break
        q = caja_desde_homografia(r["H"], tpl_bgr.shape)
        if not cuadrilatero_valido(q, tpl_bgr.shape):
            break
        r["quad"] = q
        salida.append(r)
        if not r["usados"]:
            break
        usados |= r["usados"]
    return salida


# ===========================================================================
# 7. Umbrales automaticos de deteccion
# ===========================================================================
def umbral_otsu(valores):
    """Otsu sobre los scores ya suprimidos por NMS.

    No se umbrala el mapa entero sino la poblacion de picos: ahi la separacion
    entre detecciones reales y respuestas de fondo es bimodal y Otsu la
    encuentra sin que nadie fije un numero. Es el criterio de binarizacion de
    la clase 2 aplicado al espacio de respuestas.
    """
    v = np.asarray(valores, np.float64)
    if v.size < 3:
        return -np.inf
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return -np.inf
    q = np.clip(((v - lo) / (hi - lo) * 255).astype(np.uint8), 0, 255)
    t, _ = cv.threshold(q, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return lo + (t / 255.0) * (hi - lo)


def cota_de_ruido(n_ventanas):
    """Cuanto puede valer el maximo de n_ventanas muestras de puro ruido.

    Para n gaussianas independientes el maximo esperado crece como
    sqrt(2*ln n). Exigir que un pico supere esa cota es pedir que no se explique
    como el maximo del fondo. No tiene parametros libres: depende solo de
    cuantas ventanas se miraron.
    """
    return float(np.sqrt(2.0 * np.log(max(n_ventanas, 2))))


def z_robusto(R):
    """Mapa en sigmas robustas (mediana y MAD), inmune a los propios picos."""
    med = float(np.median(R))
    mad = float(np.median(np.abs(R - med)))
    sigma = 1.4826 * mad if mad > 1e-9 else float(R.std())
    if sigma < 1e-12:
        sigma = 1.0
    return med, sigma


# ===========================================================================
# 8. API publica
# ===========================================================================
def preparar_template(tpl_bgr):
    """Todo lo que depende solo del template se calcula una sola vez."""
    recorte = recortar_template(tpl_bgr)
    return {"bgr": recorte,
            "gris": a_gris(recorte),
            "semi_trazo": semiancho_de_trazo(recorte),
            "forma": recorte.shape}


def _escalas_para(img, prep, pose):
    """Grilla fina alrededor de la escala votada por SIFT, o barrido completo."""
    tw = prep["forma"][1]
    th = prep["forma"][0]
    if pose and pose["inliers"] >= MIN_INLIERS and pose["escala"]:
        s = pose["escala"]
        esc = grilla_de_escalas(s * 0.75, s * 1.33, paso=PASO_FINO)
    else:
        esc = grilla_de_escalas(*rango_de_escalas(img.shape, prep["forma"],
                                                  prep["semi_trazo"]))
    return [s for s in esc if tw * s <= img.shape[1] and th * s <= img.shape[0]]


def _conf(valor, referencia):
    """Confianza en [0,1): 0.5 justo en el umbral de deteccion, 1 en el limite."""
    v = max(float(valor), 0.0)
    return v / (v + float(referencia)) if (v + referencia) > 0 else 0.0


def detectar(img_bgr, prep, multiples=False):
    """Detecta el logo. Devuelve lista de detecciones ordenadas por confianza.

    multiples=False -> item 1: la mejor deteccion de la imagen.
    multiples=True  -> items 2 y 3: todas las instancias.
    """
    pose = pose_sift(prep["bgr"], img_bgr)
    sift_confiable = (pose["inliers"] >= MIN_INLIERS and pose["H"] is not None
                      and cuadrilatero_valido(
                          caja_desde_homografia(pose["H"], prep["forma"]),
                          prep["forma"]))

    if sift_confiable:
        poses = (poses_sift_iterativas(prep["bgr"], img_bgr) if multiples
                 else [dict(pose, quad=caja_desde_homografia(pose["H"], prep["forma"]))])
        dets = []
        for r in poses:
            q = r["quad"]
            x0, y0 = q[:, 0].min(), q[:, 1].min()
            x1, y1 = q[:, 0].max(), q[:, 1].max()
            dets.append({"caja": (int(round(x0)), int(round(y0)),
                                  int(round(x1 - x0)), int(round(y1 - y0))),
                         "quad": q,
                         "metodo": "SIFT+RANSAC",
                         "evidencia": float(r["inliers"]),
                         "unidad": "inliers",
                         "confianza": _conf(r["inliers"], MIN_INLIERS),
                         "escala": r["escala"],
                         "polaridad": r["polaridad"]})
        if dets:
            return sorted(dets, key=lambda d: -d["confianza"])

    # --- via NCC ---------------------------------------------------------
    F = a_gris(img_bgr)
    barrido = barrido_multiescala(F, prep["gris"], _escalas_para(img_bgr, prep, pose))
    if not barrido:
        return []
    b = max(barrido, key=lambda r: r["psr"])
    R = b["R"]
    w, h = b["caja"][2], b["caja"][3]
    med, sigma = z_robusto(R)
    z_min = cota_de_ruido(R.size)

    if not multiples:
        x, y = b["caja"][0], b["caja"][1]
        z = (float(R[y, x]) - med) / sigma
        return [{"caja": (x, y, w, h), "quad": None, "metodo": "NCC multiescala",
                 "evidencia": float(b["psr"]), "unidad": "PSR",
                 "confianza": _conf(z, z_min), "escala": b["escala"],
                 "polaridad": b["polaridad"]}]

    ys, xs = np.nonzero(R > R.mean())
    if ys.size == 0:
        return []
    vals = R[ys, xs]
    orden = np.argsort(-vals)[:6000]
    cand = [{"score": float(vals[i]), "caja": (int(xs[i]), int(ys[i]), w, h)}
            for i in orden]
    picos = nms(cand)
    sc = np.array([d["score"] for d in picos])
    u = umbral_otsu(sc)
    z = (sc - med) / sigma
    ok = (sc >= u) & (z >= z_min)
    return [{"caja": picos[i]["caja"], "quad": None, "metodo": "NCC multiescala",
             "evidencia": float(z[i]), "unidad": "sigmas",
             "confianza": _conf(z[i], z_min), "escala": b["escala"],
             "polaridad": b["polaridad"]}
            for i in np.nonzero(ok)[0]]


# ===========================================================================
# 9. Visualizacion
# ===========================================================================
def dibujar(img_bgr, detecciones, color=(0, 220, 0)):
    """Dibuja cajas (orientadas si las hay) con el nivel de confianza."""
    out = img_bgr.copy()
    gro = max(2, int(round(img_bgr.shape[1] / 400.0)))
    fs = max(0.45, img_bgr.shape[1] / 1400.0)
    for d in detecciones:
        x, y, w, h = d["caja"]
        if d.get("quad") is not None:
            cv.polylines(out, [d["quad"].astype(np.int32)], True, color, gro)
        else:
            cv.rectangle(out, (x, y), (x + w, y + h), color, gro)
        txt = f"{d['confianza']:.2f}"
        (tw_, th_), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, fs, gro)
        yt = y - 4 if y - th_ - 6 >= 0 else y + h + th_ + 4
        cv.rectangle(out, (x, yt - th_ - 3), (x + tw_ + 4, yt + 3), color, -1)
        cv.putText(out, txt, (x + 2, yt), cv.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), gro)
    return out
