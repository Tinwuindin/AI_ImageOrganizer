# AI_ImageOrganizer

Una ia que me ordena las fotos que tome y solo abandone tomandas lol

Asistente de IA **local** para organizar miles de imágenes: las analiza con la GPU,
las clasifica, permite buscarlas en lenguaje natural y las acomoda en carpetas. Todo
corre en tu equipo: ninguna imagen sale de él y, una vez descargados los modelos,
funciona sin internet.

## Qué hace

- **Analiza** carpetas completas: JPEG, PNG, WebP, HEIC, TIFF, BMP y GIF.
- **Clasifica** cada imagen con categoría, subcategoría, descripción, etiquetas y
  texto visible (OCR básico).
- **Busca** por significado, en español: «mi perro en la playa», «recibos de gasolina».
- **Detecta duplicados exactos**, aunque tengan otro nombre.
- **Propone una organización** en carpetas que revisas antes de aplicarla.
- **Mueve los archivos** sin sobrescribir ni borrar nada, y puede deshacerlo.

## Requisitos

| | |
|---|---|
| Sistema | Windows 10 u 11 |
| Python | 3.11 |
| GPU | NVIDIA con 8 GB de VRAM (probado en RTX 4060) y controlador compatible con CUDA 13 |
| Disco | Unos 12 GB: 6 GB de modelos, 1 GB de llama.cpp y 5 GB de PyTorch |
| Espacio de datos | Unos 300 MB por cada 10,000 imágenes |

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py
```

El último comando descarga una sola vez, a `models/` y `tools/`, versiones fijas de los
modelos y de `llama-server`, y verifica cada archivo por SHA-256. Los dos modelos son
públicos y con licencia Apache 2.0, así que no necesitas cuenta en Hugging Face.

## Uso

Haz doble clic en `Iniciar.bat`, o ejecuta `.venv\Scripts\python run.py`. Se abre
<http://127.0.0.1:8765> en tu navegador.

1. **Análisis:** elige tus carpetas y pulsa *Analizar*. Puedes cancelar cuando quieras:
   al volver a analizar, se retoma donde quedó.
2. **Biblioteca:** explora por categorías, busca y corrige lo que haga falta.
3. **Organizar:** elige un destino y una estructura, revisa la propuesta y confirma.
4. **Duplicados:** elige qué copia conservar en cada grupo.
5. **Categorías:** adapta la lista a tus fotos. Conviene hacerlo *antes* de analizar,
   porque los cambios solo aplican a las imágenes que aún no se han clasificado.

## Cómo funciona

```
Escaneo → Duplicados → Vectores → Agrupación → Clasificación → Organización
```

| Etapa | Qué ocurre | Dónde corre |
|---|---|---|
| Escaneo | Recorrido recursivo, SHA-256, EXIF, miniaturas | CPU |
| Duplicados | Las copias idénticas no pasan por los modelos | CPU |
| Vectores | EmbeddingGemma 2 convierte cada imagen en un vector | GPU |
| Agrupación | PCA + HDBSCAN reúnen las imágenes parecidas | CPU |
| Clasificación | Gemma 4 E2B analiza representantes y casos ambiguos | GPU |
| Organización | Propuesta, revisión y movimientos con registro | CPU |

**Optimización clave:** no hace falta enviar todas las imágenes a Gemma. De cada grupo
se analizan unos representantes repartidos por todo el grupo. Si coinciden, sus miembros
heredan la clasificación del más parecido. Se analizan una por una las imágenes sin
grupo, las de grupos sin acuerdo y las que quedan lejos del centro de su grupo. Una
imagen heredada no tiene descripción propia, porque la de otra foto sería falsa.

**Los modelos se turnan la GPU:** EmbeddingGemma se descarga antes de iniciar Gemma.
La búsqueda codifica el texto en la CPU, así que funciona aunque Gemma esté trabajando.

### Decisiones medidas

Estas decisiones salieron de pruebas con fotos reales y están explicadas en el código:

- **Gemma corre en `llama-server`** con el GGUF Q4 oficial de Google. Pesa 4.3 GB frente a
  10.2 GB de los pesos completos, limita la salida con un esquema JSON y, al cerrar el
  proceso, libera toda la VRAM.
- **Importa el orden de los campos y de las categorías.** El modelo describe la imagen y
  nombra lo que ve antes de elegir la categoría, y la lista va de lo específico a lo
  genérico. Solo con eso, las capturas de pantalla pasaron de 15/30 a 23/30 aciertos.
- **La confianza es la probabilidad real del modelo**, no un número que él declara. Al
  preguntarle, respondía 0.95 acertara o no. Ahora la media es 0.96 cuando acierta y
  0.61 cuando falla, y el umbral por defecto marca para revisar el 68 % de los errores y
  solo el 5 % de los aciertos.
- **La búsqueda no recorta resultados con un umbral:** al probarlo, cuando un tema dominaba
  la biblioteca se perdía hasta el 85 % de lo relevante.

## Rendimiento medido

En una RTX 4060 Laptop de 8 GB con Ryzen 7 7735HS y 32 GB de RAM.

| Etapa | Medido | Para 10,000 imágenes |
|---|---|---|
| Escaneo inicial | 10,200 archivos de 300 KB en 50 s | ~1 min, más con fotos pesadas o disco duro |
| Escaneo repetido | 1 s | 1 s |
| Vectores | 6–8 img/s | ~25 min |
| Agrupación | 2.7 s | 3 s |
| Clasificación | 1.5 s por imagen analizada | ~2 h con grupos, ~4.2 h sin ellos |
| Búsqueda | ~100 ms; la primera tarda unos 4 s | Igual |
| Mover archivos en un mismo disco | 9,234 archivos en 15 s | 16 s |

**VRAM:** 1.6–2.2 GB con EmbeddingGemma y unos 3 GB con Gemma, nunca a la vez.

**Calidad:** 96.5 % de aciertos (251 de 260) en fotos públicas de ocho temas. El punto más
débil son las capturas de pantalla (23 de 30): las páginas con mucho texto a veces caen en
*Documentos*. Con los grupos, Gemma hizo un 55 % menos de llamadas y las 153 clasificaciones
heredadas que se pudieron comprobar fueron correctas. Estas cifras vienen de un conjunto de
prueba con temas muy separados: en una fototeca personal espera menos ahorro o más casos por
revisar. Compruébalo con una carpeta pequeña antes de procesar todo.

## Seguridad de tus archivos

- **Nada se mueve sin tu confirmación.** La propuesta no toca el disco.
- **Nunca se sobrescribe ni se borra un archivo.** Los duplicados van a `_Duplicados`, y
  borrarlos es decisión tuya.
- **Todo se puede deshacer** desde el historial.
- **Entre discos distintos,** primero se copia y se verifica el SHA-256; el original se retira
  después.
- **Las imágenes de baja confianza quedan desmarcadas** en la propuesta, y las que no se pudieron
  leer no se mueven.
- **Un cierre inesperado no pierde trabajo:** el avance se guarda por lotes y, al volver a abrir,
  se completa el registro de cualquier movimiento a medias.
- **La API solo acepta peticiones locales.** Además de escuchar en `127.0.0.1`, valida `Host` y
  `Origin`, para que una página web abierta en tu navegador no pueda enviarle órdenes.

## Configuración

Todo está en [config.yaml](config.yaml), comentado. Los ajustes más útiles:

| Ajuste | Para qué |
|---|---|
| `classification.use_clusters` | `false` analiza cada imagen con Gemma: mejores descripciones, el doble de tiempo |
| `organization.min_confidence` | Sube o baja cuántas imágenes quedan por revisar |
| `organization.min_subcategory_size` | Subcategorías menores se agrupan en `Varios` |
| `embedding.image_tokens`, `gemma.image_tokens` | Menos tokens es más rápido; más tokens leen mejor el texto |

Para cambiar de modelo, ajusta las rutas de `gemma` y `embedding`. Solo
[app/models/](app/models/) conoce los modelos; el resto del sistema no cambia.

## Limitaciones

- La detección de duplicados es exacta: una foto recomprimida o redimensionada no cuenta como
  duplicado, aunque aparece en *Imágenes parecidas*.
- Los archivos asociados (`.xmp`, `.aae`, vídeos de Live Photo) no se mueven con su imagen.
- El OCR es básico. Para documentos con letra pequeña, sube `gemma.image_tokens` a 560 o 1120.
- La búsqueda siempre muestra lo más parecido, aunque no exista nada realmente relacionado.

## Estructura

```
app/
├── api/        Rutas de FastAPI
├── core/       Configuración, escáner, metadatos, hash y miniaturas
├── models/     EmbeddingGemma 2 y Gemma 4 (lo único que conoce los modelos)
├── engine/     Pipeline, agrupación, similitud, duplicados y categorización
├── database/   SQLite (registro principal) y FAISS (vectores)
└── services/   Organización, búsqueda y tareas en segundo plano
frontend/       Interfaz web sin dependencias ni compilación
scripts/        Descarga de modelos
tests/          Pruebas automáticas
```

## Pruebas

```powershell
.venv\Scripts\python -m pytest tests
```

No necesitan GPU ni modelos: usan imágenes reales en disco con modelos simulados.
