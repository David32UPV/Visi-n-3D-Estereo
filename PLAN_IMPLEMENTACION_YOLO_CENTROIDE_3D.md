# Plan de implementación: YOLO + centroide + profundidad 3D

Objetivo: al abrir la cámara, detectar automáticamente las cajas con `best.pt` y, para cada detección, dibujar:

- la caja/hitbox,
- la clase,
- el centroide,
- y la estimación 3D aproximada del objeto usando la disparidad estéreo.

La idea es aprovechar lo que ya existe en `modules/yolo_stereo_module.py` y `modules/stereo_module.py` sin rehacer el pipeline completo.

## Hipótesis técnica

Sí, se puede sacar el centroide.

Como estás usando YOLOv8-seg, hay dos opciones:

1. Centroide de la caja delimitadora: rápido y simple, usando el centro del `bbox`.
2. Centroide de la máscara: más preciso, usando la segmentación de YOLO.

Para cajas físicas en el suelo, la opción recomendada es la segunda si la máscara es estable. Si la máscara falla o viene vacía, se puede caer al centro del `bbox`.

## Cómo calcular la profundidad

El flujo correcto sería:

1. Rectificar el par estéreo.
2. Ejecutar YOLO sobre ambas vistas o, mejor, sobre una vista y usar la otra para estimar profundidad del mismo punto.
3. Obtener el centroide del objeto en coordenadas de píxel.
4. Leer la disparidad en ese píxel o en una pequeña vecindad.
5. Convertir esa disparidad a 3D.

Con el módulo actual, lo más limpio es usar `StereoTriangulator`:

$$Z = \frac{f \cdot B}{d}$$

Donde:

- $f$ es la focal efectiva,
- $B$ es la línea base entre cámaras,
- $d$ es la disparidad.

Pero mejor aún: como ya tienes la matriz `Q` en `stereo_module.py`, puedes obtener `X/Y/Z` con `cv2.reprojectImageTo3D`, que evita hacer la matemática manual y reduce errores de signo o escala.

## Plan de implementación propuesto

### Fase 1. Extraer centroides desde YOLO

Modificar `YoloSegBoxModule` para que, además de la imagen anotada, devuelva por detección:

- clase,
- confianza,
- `bbox` (`x1, y1, x2, y2`),
- centroide `(cx, cy)`,
- máscara si existe.

Regla de centroide:

- si hay máscara, calcular centroide de la máscara,
- si no hay máscara, usar el centro del `bbox`.

Resultado esperado: una lista de detecciones estructuradas, no solo la imagen pintada.

### Fase 2. Alinear centroides con estéreo

En `StereoTriangulator`:

- rectificar ambas imágenes,
- calcular disparidad sobre el par rectificado,
- para cada centroide, convertir su posición a coordenadas rectificadas,
- buscar la disparidad en ese punto.

Si el centroide cae en un píxel inválido o ruidoso:

- tomar una ventana pequeña alrededor del centroide,
- usar la mediana de las disparidades válidas,
- o, si la máscara está disponible, usar solo los píxeles del interior de la máscara.

Esto es importante porque el centro geométrico de una caja no siempre coincide con una región de disparidad limpia.

### Fase 3. Obtener 3D por objeto

Extender `StereoTriangulator` con un método tipo:

- `get_3d_from_pixel(disparity, u, v)` para un punto concreto,
- o un método nuevo por objeto, por ejemplo `get_object_3d(...)`.

Salida por detección:

- `X`, `Y`, `Z` en la unidad de la calibración,
- disparidad usada,
- estado de validez.

Recomendación: reutilizar el método ya existente `get_3d_from_pixel` en `modules/stereo_module.py` y, si hace falta, ampliarlo para soportar promedio de vecindad o centroides de máscara.

### Fase 4. Dibujar overlay enriquecido

En el overlay de YOLO, dibujar encima de cada objeto:

- caja,
- clase,
- confianza,
- punto del centroide,
- texto con `X/Y/Z` o solo `Z` si quieres simplificar.

Formato sugerido:

- `caja_1 0.91`
- `centro: (512, 384)`
- `Z: 1240 mm`

### Fase 5. Integración en `main.py`

Hacer que el modo YOLO de arranque:

- cargue `best.pt` automáticamente,
- rectifique,
- calcule disparidad,
- dibuje detecciones,
- y añada el centroide/profundidad en cada frame.

Así no habrá que pulsar nada: al abrir la cámara, el modo YOLO será el activo por defecto.

## Cambios concretos por archivo

### `modules/yolo_stereo_module.py`

- Añadir una estructura de salida por detección.
- Exponer `boxes`, `classes`, `confidences`, `centroids` y, si existe, `masks`.
- Añadir una función auxiliar para calcular centroide de máscara o de bbox.

### `modules/stereo_module.py`

- Reutilizar `get_3d_from_pixel`.
- Opcionalmente añadir un método para estimar profundidad robusta en una vecindad del centroide.
- Si se quiere máxima robustez, añadir una función que use la máscara de YOLO como región de muestreo.

### `main.py`

- Mantener el arranque automático en modo YOLO.
- Mostrar centroides y profundidad en el overlay.
- Si la estimación de profundidad falla, seguir mostrando la caja y la clase, pero marcar `Z` como inválida.

## Riesgos y decisiones

1. Si la disparidad es ruidosa, el centroide puede caer en una zona inválida. Solución: ventana local o máscara completa.
2. Si el objeto está muy cerca o con oclusión, la profundidad puede ser inestable. Solución: suavizado temporal o mediana de varios frames.
3. Si la máscara YOLO no es buena, el centroide por máscara puede ser peor que el del `bbox`. Solución: fallback automático.

## Orden recomendado de implementación

1. Primero sacar `centroides` desde YOLO y dibujarlos.
2. Después calcular `Z` desde disparidad en el centroide.
3. Luego añadir `X/Y/Z` completos con `Q`.
4. Por último, robustecer con ventana local o máscara.

## Criterio de éxito

Se considerará listo cuando, al arrancar la aplicación:

- se vean las dos lentes,
- YOLO detecte automáticamente las cajas,
- se dibuje clase + hitbox + centroide,
- y se muestre una estimación de profundidad por objeto sin pulsar ninguna tecla.
