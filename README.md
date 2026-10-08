# TP3 — Detección del logotipo de Coca-Cola

**Visión por Computadora I — CEIA, FIUBA**

Detección de un logotipo dentro de siete imágenes a partir de un único template, con **todos los
parámetros derivados automáticamente** del par (template, imagen).

Todo el trabajo está en el notebook **`TP3_VPC1.ipynb`**, que ya tiene las salidas ejecutadas. El
algoritmo vive en **`detector.py`**, separado del notebook porque la consigna pide un algoritmo
pulido que reciba las imágenes y devuelva las detecciones, no un cuaderno de exploración.

## Consigna

> Encontrar el logotipo de la gaseosa dentro de las imágenes provistas en `Material_TPs/TP3/images`
> a partir del template `Material_TPs/TP3/template`.
>
> 1. **(4 puntos)** Obtener una detección del logo **en cada imagen** sin falsos positivos.
> 2. **(4 puntos)** Plantear y validar un algoritmo para **múltiples detecciones** en la imagen
>    `coca_multi.png`, con el **mismo témplate** del ítem 1.
> 3. **(2 puntos)** **Generalizar** el algoritmo del ítem 2 **para todas las imágenes**.
>
> Visualizar los resultados con **bounding boxes** en cada imagen mostrando el **nivel de
> confianza** de la detección.

## Resultados

| Ítem | Pedido | Obtenido |
|---|---|---|
| 1 | una detección por imagen, sin falsos positivos | **7/7 correctas, 0 falsos positivos** |
| 2 | múltiples detecciones en `coca_multi.png` | **17 de 17 logos completos, 0 falsos positivos** |
| 3 | generalizar a todas las imágenes | la misma llamada corre sobre las 7 |

En `coca_multi.png` hay además **4 logos cortados por el borde del cuadro** que no se detectan, y no
pueden detectarse: la correlación necesita la ventana del template entera dentro de la imagen.

## Los cuatro problemas que había que resolver

1. **Polaridad invertida.** El template es oscuro sobre claro, pero en seis de las siete imágenes el
   logo aparece claro sobre oscuro. La correlación entre un patrón y su inverso vale −1, así que un
   detector ingenuo no encuentra nada. Se resuelve con **|NCC|**, que es invariante al signo; el
   signo del pico informa la polaridad como subproducto. La única imagen con la polaridad del
   template es `coca_retro_1.png`, que además es escala de grises pura (saturación media 0.0), con
   lo cual **ninguna heurística de color podía usarse**.

2. **NCC degenerada.** `cv.TM_CCOEFF_NORMED` divide por la desviación de la ventana: sobre fondo
   plano el cociente es 0/0 y OpenCV devuelve **1.0**, de modo que el máximo cae sobre el fondo
   blanco. Se corrige con un **piso de varianza atado a la energía del template**.

3. **Escalas no comparables.** El máximo crudo de la NCC favorece sistemáticamente la escala más
   chica. Se usa el **peak-to-sidelobe ratio**, que normaliza cada respuesta contra su propia
   distribución y sí es comparable entre escalas.

4. **Márgenes del template.** El archivo mide 400×175 (aspecto 2.29) pero el logo ocupa 373×115
   (aspecto 3.24): los márgenes blancos deformaban el aspecto un **41 %**. Se recorta
   automáticamente por Otsu sobre la distancia al color de fondo.

## Arquitectura

Dos detectores complementarios y una regla de fusión automática:

- **A — SIFT + ratio test de Lowe + homografía por RANSAC.** Invariante a escala, rotación y
  perspectiva; es el único que tolera la deformación cilíndrica de una lata. Da caja orientada y una
  estimación de escala por votación de keypoints (Hough generalizado de Lowe).
- **B — Correlación normalizada multiescala sobre niveles de gris.** Cubre instancias chicas y
  repetidas, donde SIFT se queda sin keypoints.

```
si SIFT tiene >= 8 inliers y su cuadrilátero es geométricamente válido:
        lo localiza SIFT
en otro caso:
        lo localiza la NCC
```

El umbral de 8 es el doble del mínimo algebraico de 4 que exige una homografía.

**Por qué hacen falta los dos:** `coca_logo_2.png` es una lata cilíndrica y la NCC rectangular se va
al texto recto «PLUS CAFÉ ESPRESSO»; `coca_multi.png` tiene veinte logos de ~75 px y SIFT se queda
en 4 inliers, el mínimo algebraico, que no es evidencia de nada. **Las dos únicas imágenes donde los
detectores discrepan son exactamente las dos que motivaron la fusión.**

## Parámetros automáticos

| Qué | De dónde sale |
|---|---|
| escala máxima | el template tiene que entrar en la imagen |
| escala mínima | semiancho de trazo del template (transformada de distancia) → ~67 px |
| polaridad | signo del pico de la NCC / inliers de RANSAC |
| escala del objeto | votación de keypoints inliers, o máximo del PSR |
| umbral de detección | Otsu sobre los picos **y** cota de ruido √(2·ln n) |
| tamaño de la caja | escala seleccionada × tamaño del template |

Quedan cinco constantes **adimensionales y globales**, ninguna ajustada por imagen: `PISO_VAR`
(0.01), `IOU_NMS` (0.30), `RATIO_LOWE` (0.80, del paper de Lowe), `MIN_INLIERS` (8) y `PASO_ESCALA`
(1.06). La sección 13 del notebook mide la sensibilidad a las dos únicas elegidas por criterio
propio: el resultado no cambia.

## Cómo correrlo

```bash
pip install -r requirements.txt

# reconstruir y ejecutar el notebook completo
python build_nb.py

# calificar la entrega
python calificador.py
```

Para usar el detector sobre una imagen cualquiera:

```python
import cv2 as cv, detector as D

prep = D.preparar_template(cv.imread('data/template/pattern.png', cv.IMREAD_COLOR))
img  = cv.imread('data/images/coca_multi.png', cv.IMREAD_COLOR)

dets = D.detectar(img, prep, multiples=True)
cv.imwrite('salida.png', D.dibujar(img, dets))
```

Cada detección es un diccionario con `caja` (x, y, w, h), `quad` (caja orientada, si vino de SIFT),
`confianza` ∈ [0,1), `metodo`, `evidencia`, `unidad`, `escala` y `polaridad`.

## Estructura

```
TP3_VPC1.ipynb          notebook entregable, con las salidas ejecutadas
detector.py             el algoritmo
calificador.py          agente calificador contra la rúbrica del enunciado
build_nb.py             arma y ejecuta el notebook desde cells_tp3.py
cells_tp3.py            contenido de las celdas
data/images/            las 7 imágenes provistas
data/template/          el template provisto
data/ground_truth.json  verdad de terreno para el test de regresión
requirements.txt
```

## Sobre la verdad de terreno

`data/ground_truth.json` se anotó **por inspección visual** de cada imagen, revisando las cajas una
por una sobre la figura ampliada, y el conteo de `coca_multi.png` contando a mano los logos
completos de los dos estantes (9 arriba + 8 abajo = 17). Sirve como **test de regresión** del
detector, no como benchmark independiente: está declarado así dentro del propio archivo.

## Limitaciones conocidas

1. **Los 4 logos cortados por el borde de `coca_multi.png`.** Estructural: la correlación necesita
   la ventana completa dentro de la imagen.
2. **El logo del vasito en `coca_retro_2.png`**, de unos 30 px, queda por debajo de los 67 px que
   fija el ancho de trazo del template. Haría falta un template de mayor resolución, no otro
   parámetro.
3. **Las tres botellas desenfocadas de `logo_1.png`** quedan justo debajo de la cota de ruido (~4 σ
   contra las 4.9 que exige la cota). El desenfoque destruye las altas frecuencias, que son
   justamente las que la correlación usa para distinguir el logo del fondo — el mismo fenómeno que
   medía la métrica de enfoque del TP2.

## Entorno

Python 3.10, OpenCV 4.8.1, NumPy 1.26.2. Versiones exactas en `requirements.txt`.
