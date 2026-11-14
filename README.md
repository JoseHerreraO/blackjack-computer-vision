# Blackjack Computer Vision 🃏🕶️

Asistente de blackjack basado en **visión por ordenador clásica**. El jugador lleva unas **gafas Meta** cuya cámara captura la mesa en primera persona. El móvil recibe el vídeo de las gafas a través del SDK de Meta (Wearables Device Access Toolkit) y lo reenvía por la red local al ordenador, que es donde se hace todo el análisis. El ordenador reconoce las cartas del jugador y del dealer, recomienda la jugada óptima según la estrategia básica (pedir, plantarse, doblar o separar) y lo muestra en un HUD sobre el vídeo, con conteo Hi-Lo, cartas restantes en el zapato y un historial de decisiones.

Antes de usar el asistente hay que desbloquearlo enseñando a la cámara **cuatro ases en un orden secreto de palos**.

Proyecto final de *Visión por Ordenador I* en 3º de Ingeniería Matemática, Universidad Pontificia Comillas ICAI (curso 2026/27).

**Autores.** José De Oña · José Herrera (Grupo A)

> 🚧 **Estado.** El repositorio tiene la estructura del paquete, los tests y la CI. Los módulos están en desarrollo (ver [Planificación](#planificación)).

---

## Arquitectura

```
┌──────────────┐  Bluetooth (SDK de Meta)  ┌──────────────┐  Wi-Fi local (frames)  ┌───────────────────────┐
│  Gafas Meta  │ ────────────────────────► │  App móvil   │ ─────────────────────► │  Ordenador (Python)   │
│  (cámara)    │                           │  Android/iOS │                        │  Receptor de frames   │
└──────────────┘                           └──────────────┘                        └───────────┬───────────┘
                                                                                               │
[Calibración offline] ──► K, coeficientes de distorsión, RMS                                   │
                                   │                                                           ▼
                                   └──────────► Corrección de distorsión (remap) ──► Detector de cartas
                                                                                          │
   ┌──────────────── SISTEMA DE SEGURIDAD ────────────────────────────────────────────────┴───┐
   │  Detector de palo del as ──► Decodificador de secuencia (4)                              │
   └───────────────────────────────┬──────────────────────────────────────────────────────────┘
                                   │ acceso concedido
   ┌──────────────── SISTEMA PROPUESTO ───────────────────────────────────────────────────────┐
   │  Tracker (cartas + fichas) ──► Reconocimiento rango/palo                                 │
   │  ──► Lógica de ronda ──► Estrategia básica + Hi-Lo                                       │
   │  ──► Salida de vídeo (HUD + FPS)                                                         │
   └──────────────────────────────────────────────────────────────────────────────────────────┘
```

El móvil no hace ningún procesamiento de visión. Solo actúa de puente entre las gafas y el ordenador, que concentra todo el pipeline en Python con OpenCV.

| Módulo | Técnica |
|---|---|
| Captura | Cámara de las gafas Meta mediante el Wearables Device Access Toolkit (máximo 720p a 30 FPS por el ancho de banda de Bluetooth). La app móvil reenvía los frames al ordenador por la red local. Plan B con webcam cenital fija conectada al ordenador |
| Calibración | Tablero de ajedrez fotografiado con las propias gafas y a la misma resolución del stream. `findChessboardCorners` + `cornerSubPix` + `calibrateCamera` (offline). Mapas de `initUndistortRectifyMap` precalculados |
| Detección de cartas | Otsu / HSV, morfología, `findContours`, `approxPolyDP` (4 vértices, aspecto ≈ 1.4), homografía a 200×300 |
| Seguridad | Palo del pip central del as por color (HSV) y forma (momentos de Hu). Decodificador de 4 patrones con reset por error, timeout o tecla |
| Reconocimiento | Bolsa de palabras visuales. SIFT sobre la carta rectificada, vocabulario con K-Means, histograma + SVM (kernel χ²). Filtro por color del palo y votación temporal sobre N frames |
| Tracker | IDs por asociación de centroides + filtro de Kalman (cartas), CamShift (fichas) |
| Juego | Máquina de estados de ronda y tablas de estrategia básica (duras, blandas, parejas) |
| Ampliaciones | Hi-Lo (cuenta corriente y verdadera, ventaja, apuesta), histograma del zapato, log CSV, estudio de vocabulario/kernel y comparación BoW vs. template matching |

### Consecuencias de usar una cámara en primera persona

La cámara ya no está fija sobre la mesa sino en la cabeza del jugador, así que la imagen se mueve y cambia de perspectiva. La homografía de cada carta corrige la perspectiva de forma individual, pero el tracker tiene que tolerar el movimiento de la cámara entre frames y no solo el de las cartas. La resolución y los FPS de entrada están limitados por el stream de las gafas, que además los reduce automáticamente si el Bluetooth va justo. La calibración tiene que hacerse con imágenes capturadas por las gafas a la resolución final del stream, porque cambiar de resolución invalida la matriz intrínseca.

### Plan B con cámara cenital fija

Como respaldo, el sistema también funciona con una webcam (o un móvil) fija en posición cenital sobre la mesa y conectada directamente al ordenador. En ese modo se salta el puente gafas → móvil y el resto del pipeline es idéntico, así que solo cambia la fuente de frames. Sirve para desarrollar y probar sin depender de las gafas y para la demo si el stream de Bluetooth falla o baja de calidad. Con la cámara fija el tracker lo tiene más fácil, porque solo se mueven las cartas y las fichas. La calibración se hace por separado para cada cámara y se guarda en `data/calibration/` con parámetros distintos para las gafas y para la webcam.

## Estructura del repositorio

```
.
├── src/blackjack_vision/
│   ├── calibration.py     # Calibración offline con tablero de ajedrez
│   ├── undistort.py       # Corrección de distorsión (initUndistortRectifyMap + remap)
│   ├── card_detector.py   # Umbralizado, contornos cuadriláteros y rectificación
│   ├── recognition.py     # Reconocimiento de rango y palo con bolsa de palabras
│   ├── security.py        # Detector de palo del as y decodificador de la secuencia
│   ├── tracker.py         # Seguimiento de cartas y fichas
│   ├── strategy.py        # Lógica de ronda, estrategia básica y Hi-Lo
│   └── hud.py             # HUD con recomendación, conteo, historial y FPS
├── scripts/
│   ├── calibrate.py       # Ejecuta la calibración y guarda los parámetros
│   └── run.py             # Lanza el asistente en tiempo real
├── tests/                 # Tests con pytest
├── data/
│   ├── calibration/       # Imágenes del tablero y parámetros de calibración
│   └── samples/           # Fotos y vídeos de prueba (no se suben al repo)
├── .github/               # CI (ruff + pytest) y plantilla de pull request
├── pyproject.toml
└── requirements.txt
```

Faltan por añadir el receptor de frames del lado del ordenador y la app móvil que usa el SDK de Meta.

## Instalación y uso

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

```bash
python scripts/calibrate.py     # calibración offline de la cámara de las gafas
python scripts/run.py           # asistente en tiempo real
```

Para comprobar el código antes de abrir una pull request se ejecutan los mismos pasos que la CI.

```bash
ruff check .
pytest
```

## Material

Unas gafas Meta compatibles con el Wearables Device Access Toolkit, un móvil Android o iOS con la app puente, un ordenador en la misma red Wi-Fi, una baraja francesa estándar, fichas de colores y un tapete de color uniforme. Para el plan B hace falta además una webcam (o un móvil) con soporte para montarla en posición cenital sobre la mesa.

## Planificación

| Fechas | Tareas | Estado |
|---|---|---|
| 4–7 nov | Repositorio, lectura de cámara y propuesta | ✅ |
| 8–14 nov | Estructura del paquete y CI. Calibración, detector y reconocimiento de cartas | ⏳ |
| 15–20 nov | App puente gafas → móvil → ordenador. Sistema de seguridad, lógica de ronda y estrategia básica | ⏳ |
| 21–26 nov | Tracker, Hi-Lo, zapato, log, HUD y optimización | ⏳ |
| 27–30 nov | Informe, vídeo demo y póster | ⏳ |
| 2 dic | Entrega y presentación | ⏳ |

## Licencia

[MIT](LICENSE)
