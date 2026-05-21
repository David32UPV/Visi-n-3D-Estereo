# Plan de implementacion: dataset Roboflow + YOLOv8 + ZED 2

## Objetivo

Quiero añadir una unica pieza nueva al proyecto para que haga dos cosas:

1. Entrenar un modelo YOLOv8 de Ultralytics con tu dataset de Roboflow.
2. Usar ese modelo en tiempo real sobre las dos lentes de la ZED 2 para detectar automaticamente `caja_1` y `caja_2`.

Como tu etiquetado es por poligonos, la base correcta es **YOLOv8-seg**. Eso nos permite respetar las mascaras que has dibujado y, si luego solo queremos cajas delimitadoras, convertir internamente la salida a bounding boxes sin cambiar el flujo principal.

## Que tiene este directorio del dataset y por que existen esos archivos

Segun lo que has descargado, la estructura de Roboflow sirve para dejar todo listo para entrenar sin tener que reorganizar nada manualmente.

### `data.yaml`

Es el archivo de configuracion principal del dataset. Contiene:

- La ruta a `train`, `val` y `test`.
- El numero de clases (`nc: 2`).
- Los nombres de clase (`caja_1`, `caja_2`).
- Metadatos de Roboflow del proyecto y la version.

Este archivo es el que usaremos para decirle a Ultralytics donde esta el dataset y como se llaman las clases.

### `README.dataset.txt`

Es una descripcion automatica del dataset exportado.
Normalmente incluye:

- Cuantas imagenes tiene.
- Que preprocesado se aplico.
- Si se hizo resize, orientacion automatica o aumentos.

No hace falta para entrenar, pero sirve para saber exactamente que contiene la exportacion.

### `README.roboflow.txt`

Es el texto informativo de Roboflow.
Explica de donde viene el dataset, cuando se exporto y en que formato se genero.
Tampoco es necesario para el entrenamiento, pero documenta el origen del dataset.

### `train/images/`

Aqui estan las imagenes de entrenamiento.
Cada imagen debe tener su archivo de etiqueta con el mismo nombre base en `train/labels/`.

### `train/labels/`

Aqui estan las anotaciones asociadas a cada imagen.
Como tu etiquetado es por poligonos, estos ficheros son los que almacenan la geometria de cada objeto.

En una exportacion YOLOv8-seg, cada linea representa una instancia de objeto con su clase y sus puntos de segmento.

### Sobre `val` y `test`

Tu `data.yaml` ya deja preparadas las rutas `val` y `test`, aunque en la carpeta que muestras solo aparece `train`.
Eso suele pasar cuando:

- El export de Roboflow deja las rutas previstas, pero no has descargado la particion valid/test.
- O el proyecto generara la validacion de forma automatica durante el entrenamiento.

En el plan que propongo, si no existen esas particiones, el script las crea a partir de `train` de forma automatica para no obligarte a preparar nada a mano.

## Estructura minima que quiero implementar

Voy a crear un modulo nuevo en `modules/` para no mezclar el entrenamiento/inferencia con el resto del pipeline.

Propuesta de archivo:

- `modules/yolo_stereo_module.py`

La idea es que este archivo concentre lo esencial:

- cargar el dataset,
- entrenar YOLOv8-seg,
- ejecutar inferencia sobre la imagen izquierda y derecha,
- dibujar resultados,
- y dejar listo el siguiente paso si quieres fusionar detecciones entre ambas lentes.

## Funciones que quiero implementar

### 1. `__init__`

Responsabilidad:

- guardar rutas del dataset y de calibracion,
- elegir el modelo base de Ultralytics,
- preparar la clase para entrenamiento e inferencia,
- y mantener el codigo lo mas corto posible.

Que buscaria simplificar:

- no repetir rutas por todo el script,
- no mezclar configuracion con logica,
- y dejar una sola instancia reutilizable.

### 2. `load_dataset_config()`

Responsabilidad:

- leer `data.yaml`,
- comprobar que existen las clases y rutas esperadas,
- y detectar si faltan `val` o `test`.

Por que la necesito:

- porque Ultralytics depende mucho del YAML,
- y porque es mejor fallar pronto si el dataset esta incompleto.

### 3. `prepare_dataset_splits()`

Responsabilidad:

- crear una particion de validacion si no existe,
- y dejar el dataset listo para entrenar.

En tu caso esta funcion solo se ejecutaria si el export no trajo validacion real.

La idea es que haga algo simple:

- leer las imagenes de `train/images`,
- separar una pequena parte para `val`,
- copiar o mover las imagenes y sus labels,
- y actualizar el YAML o generar uno temporal para el entrenamiento.

Esto evita que tengas que tocar el dataset a mano.

### 4. `train_model()`

Responsabilidad:

- lanzar el entrenamiento con Ultralytics YOLOv8-seg,
- usar el `data.yaml` preparado,
- guardar pesos en `runs/` o en una carpeta que definamos,
- y devolver el modelo entrenado o la ruta del mejor checkpoint.

Que haria internamente:

- cargar un modelo base como `yolov8n-seg.pt` o `yolov8s-seg.pt`,
- entrenar con tus dos clases,
- y dejar guardado `best.pt`.

Mi intencion es empezar con un modelo pequeno para que entrene rapido y sea facil de probar con la ZED 2.

### 5. `load_trained_model()`

Responsabilidad:

- cargar el `best.pt` ya entrenado,
- comprobar que el archivo existe,
- y dejar el modelo listo para inferencia.

Esto separa claramente el entrenamiento del uso en tiempo real.

### 6. `predict_frame(frame)`

Responsabilidad:

- ejecutar YOLO sobre una sola imagen,
- obtener las detecciones,
- y devolverlas de forma simple.

Salida esperada:

- clase,
- confianza,
- mascara si existe,
- y bounding box calculada a partir de la mascara o directamente del resultado del modelo.

Esta funcion es la base del tiempo real.

### 7. `predict_stereo(frame_left, frame_right)`

Responsabilidad:

- ejecutar la red sobre la imagen izquierda y la derecha,
- dibujar las detecciones en ambas,
- y devolver ambas vistas anotadas.

Aqui solo busco deteccion visual en las dos lentes.
No mezclo 3D todavia para no complicar el primer paso.

### 8. `match_left_right_detections()`

Responsabilidad:

- emparejar detecciones de la izquierda con las de la derecha,
- usando la clase y la proximidad vertical de las cajas,
- que es lo correcto en un sistema rectificado.

Por que hace falta:

- si luego queremos calcular posicion 3D,
- necesitamos saber que deteccion izquierda corresponde a cual deteccion derecha.

### 9. `triangulate_matched_detections()`

Responsabilidad:

- calcular el centro de cada deteccion emparejada,
- triangular esos centros con tu calibracion estéreo,
- y devolver la posicion 3D de cada caja.

Esta funcion la dejaria como segunda fase, porque primero quiero asegurar deteccion correcta en cada lente.

### 10. `draw_detections()`

Responsabilidad:

- pintar cajas, clases y confianza sobre la imagen,
- de forma limpia y reutilizable.

Ventaja:

- evita repetir codigo en cada modo o script,
- y hace que el resultado en pantalla sea mas facil de leer.

## Flujo de trabajo que seguiria

### Fase 1: preparar el entrenamiento

1. Leer `data.yaml`.
2. Comprobar si hay `val`.
3. Si no existe, crear una validacion minima automaticamente.
4. Llamar al entrenamiento de YOLOv8-seg.
5. Guardar los pesos entrenados.

### Fase 2: probar deteccion en una imagen

1. Cargar `best.pt`.
2. Pasar una imagen de prueba por el modelo.
3. Dibujar la deteccion.
4. Verificar que `caja_1` y `caja_2` se distinguen bien.

### Fase 3: integrar con la ZED 2

1. Abrir la camara con `ZEDCamera`.
2. Obtener `frame_left` y `frame_right`.
3. Rectificar ambos frames con `StereoTriangulator`.
4. Ejecutar YOLO en cada lente.
5. Dibujar las detecciones en ambas ventanas.
6. Si mas adelante quieres distancia 3D, emparejar ambas detecciones y triangular sus centros.

## Integracion con el proyecto actual

Ahora mismo ya tienes:

- `modules/camera_module.py` para abrir y leer la ZED 2,
- `modules/stereo_module.py` para rectificar y triangular,
- `modules/gesture_module.py` para MediaPipe,
- y `main.py` como aplicacion interactiva.

Mi idea es no tocar demasiado eso al principio.
Primero añadiria el modulo YOLO nuevo, y luego haria una integracion minima:

- o bien un modo nuevo en `main.py`,
- o bien un script separado para probar deteccion estéreo sin romper los modos que ya tienes.

## Por que esta version es la mas simple

He evitado separar en muchos archivos porque aqui lo importante es que funcione pronto y con poco codigo.
Por eso el plan se apoya en:

- un solo modulo nuevo,
- funciones cortas y con responsabilidad unica,
- y reutilizar lo que ya existe en el proyecto para camara y rectificacion.

## Resultado esperado

Al final deberiamos tener:

- un entrenamiento YOLOv8-seg listo para tu dataset,
- un `best.pt` para detectar `caja_1` y `caja_2`,
- deteccion visual en la lente izquierda y derecha de la ZED 2,
- y la base preparada para sacar posiciones 3D si luego quieres usar ambas vistas como sistema estéreo completo.

## Siguiente paso propuesto

Si este plan te encaja, el siguiente paso sera:

1. crear `modules/yolo_stereo_module.py`,
2. añadir el entrenamiento y la inferencia minima,
3. y dejar un modo de prueba para la ZED 2.
