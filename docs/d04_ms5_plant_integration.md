# D04 MS5: выполненные изменения


## 3. Добавлены профиль MS5 и описание класса

Создан `models/d04_ms5_plant.json`: `model_type=yolox`, `input_profile=ms5`, класс `0: plant`, confidence 0.30, NMS IoU 0.45. В `DetectorSettings` добавлено `input_profile: Literal["rgb", "ms5"] = "rgb"`. В `ModelMetadata` и его загрузчик добавлено чтение этого поля. RGB остаётся прежним профилем для моделей без MS5-метаданных.

## 4. Добавлено чтение пятиканального TIFF

В `core/image_io.py` добавлены три функции:

- `read_ms5_image(path)` читает TIFF через `tifffile`, приводит YXS/YXC или SYX/CYX/QYX/IYX к `(H,W,5)`, нормализует целочисленные данные, сохраняет значения float без повторного деления.
- `validate_ms5(image)` проверяет пять каналов, ненулевые размеры, float32, отсутствие NaN/Inf и диапазон `[0,1]`.
- `ms5_rgb_preview(image)` берёт первые три полосы и формирует непрерывный RGB uint8: `(image[..., :3] * 255).clip(0,255).astype(np.uint8)`.

Формат входа реализации: **один TIFF с уже совмещёнными полосами строго `[R,G,B,RE,NIR]`**. Полосы не переставляются по предположениям, отдельные файлы автоматически не собираются. RGB JPEG и TIFF с неверным числом каналов отклоняются. Пространственные данные по-прежнему извлекаются из встроенного XMP этого TIFF; внешние файлы геопривязки не используются.

В `requirements.txt` и `pyproject.toml` добавлен `tifffile>=2023.7.10`. В тестовом окружении установлен tifffile 2025.5.10; лицензия проверена по метаданным установленного пакета — BSD-3-Clause.

## 5. Разделены данные детектора и просмотра

В `FramePayload` добавлено исключаемое из сериализации поле `detector_image`. Для MS5:

```python
payload.image           # RGB uint8 (H,W,3): просмотр, карта, RGB-классификатор
payload.detector_image  # MS5 float32 (H,W,5): D04
```

В `load_image_payload()` и источники изображений передан `input_profile`. XMP, размеры и детекции остаются в том же payload. В `BatchImageWorker` прежний `detect(payload.image)` сохранён для RGB; для MS5 используется:

```python
detector.detect(payload.detector_image, classification_image=payload.image)
```

Перед отправкой в GUI RGB копируется и помечается read-only, а `detector_image` убирается из передаваемого payload. Пятиканальный куб остаётся внутри обработки кадра.

## 6. Изменён инференс D04 и пересчёт bbox

В `BaseDetector.detect()` и `OnnxDetector.detect()` добавлен именованный аргумент `classification_image=None`. В MS5-ветке выполняется:

```python
resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
scale_x, scale_y = 640.0 / orig_w, 640.0 / orig_h
pad_x = pad_y = 0.0
```

Вместо одного коэффициента `scale` координаты по X делятся на `scale_x`, по Y — на `scale_y`. Это сохраняет правильные рамки на прямоугольных исходниках. Для RGB остаётся прежний letterbox с одинаковыми масштабами X/Y.

Существующий `_decode_yolox()` оставлен: его математический контракт соответствует фактическим трём выходам D04. При загрузке MS5 проверяются тип/форма входа и формы всех выходов; неподдерживаемый экспорт завершается явной ошибкой.

## 7. Подключена классификация RGB-кропов

Источник кропа заменён с `image[cy1:cy2, cx1:cx2]` на `classification_image[cy1:cy2, cx1:cx2]`. Сохранены отступ 10%, resize до 224×224 и ImageNet mean/std. В классификатор поступают только три RGB-канала. Проверяются совпадение геометрии RGB и MS5, RGB uint8, реальные формы входов/выходов C06/C07.

Если RGB не передан в прямом вызове `detect(ms5)`, он строится из первых трёх каналов. Отсутствующий, несовместимый или упавший классификатор теперь выдаёт ошибку, а не скрыто подменяет результат классификации классом `plant`. При отключённой классификации остаётся `plant`. Поля класса, вероятности и фазы продолжают использовать существующие отображение и экспорт.

## 8. Изменено поведение выбора моделей

В `DetectionFilterPanel` отображается короткая подпись «D04 MS5 — 5 каналов», в `userData` остаётся путь ONNX. Подписи C06/C07 заменены на «виды растений» и «категории + фазы», поскольку реальные выходы не соответствуют старым надписям «6 видов»/«7 видов».

В `OfflineController` при выборе D04 загружается профиль MS5, классификатор сбрасывается на `none`, очищаются старые детекции, статусы и карта. C06/C07 включаются отдельно. Выбранная конфигурация детектора сохраняется в `QSettings` под `offline/detector` и восстанавливается при запуске. Диалог настроек сохраняет поля классификатора и получает актуальные настройки офлайн-контроллера.

Во время пакета заблокированы модель, классификатор, очистка списка и открытие другого источника; повторный запуск пакета игнорируется. После завершения/отмены элементы снова доступны. Для MS5 запуск RGB-видео запрещён. Причина ошибки последнего кадра сохраняется в итоговом сообщении, а не затирается общим счётчиком.

## 9. Предпросмотр MS5 вынесен в поток

Создан `workers/ms5_preview_worker.py`: чтение TIFF и формирование RGB выполняются в `Ms5PreviewWorker`. Результат возвращается сигналом. Контроллер использует номер поколения предпросмотра: устаревший ответ от предыдущего выбранного файла не заменяет текущий снимок. При смене файла/модели/очистке предыдущему воркеру выставляется флаг остановки.

## 10. Миниатюры карты переведены на фоновую загрузку

В `MosaicFootprintItem` убрано синхронное `cv2.imread()` из GUI. Готовая RGB-миниатюра payload используется сразу. Если есть только путь, `ThumbnailController` запускает `ThumbnailJob` через `QThreadPool`; воркер распознаёт пятиканальный TIFF, строит RGB-миниатюру и возвращает read-only массив сигналом. Создание QPixmap остаётся в GUI.

Callback хранится как weak reference; перед обновлением проверяется существование Qt-объекта. Если снимок повреждён, остаётся контур кадра. Импорт контроллера миниатюр сделан внутри ветки загрузки, чтобы исключить цикл при самостоятельном импорте `MainWindow`.

## 11. Добавлены и выполнены проверки

`tests/test_ms5_integration.py` проверяет TIFF uint8/uint16/float32, планарную и чередующуюся укладку, порядок полос, некорректный вход, масштабирование bbox, RGB-кроп, отмену, XMP → карту → экспорт, реальные ONNX D04/C06/C07, ошибки несовместимой модели/классификатора и независимый импорт интерфейса.

Независимый расчёт на выходах реального D04 воспроизводит исходную формулу grid/exp/sigmoid, использует OpenCV NMS и сравнивает bbox/score с результатом приложения. Тесты контроллера проверяют выбор, сохранение/восстановление, фоновый просмотр, блокировку переключателей, отказ на RGB-входе. Тест карты проверяет правильные RGB-цвета фоновой MS5-миниатюры. Старые C07-тесты переведены с отсутствующего абсолютного пути на реальные установленные веса.

Команда полного прогона:

```bash
QT_QPA_PLATFORM=offscreen /tmp/geoweed-window-fix-venv/bin/python -m pytest
```

Результат: **334 passed, 1 skipped, 1 warning in 42.39s**. Единственный пропуск — отсутствующий `flight_sample.mp4`; реальные тесты D04/C06/C07 выполнены. Метрики качества классификации на размеченной мультиспектральной съёмке этим прогоном не измерялись.

## 12. Точные изменения кода

Ниже — diff фактически изменённых текстовых файлов относительно состояния до реализации. Строки с `-` заменены/удалены, строки с `+` добавлены. Для новых файлов приведён весь код.

### `pyproject.toml`

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -25,6 +25,7 @@
     "fastapi>=0.100.0",
     "uvicorn>=0.22.0",
     "Pillow>=10.0.0",
+    "tifffile>=2023.7.10",
     "paramiko>=2.9.0",
 ]
 
```

### `requirements.txt`

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -10,6 +10,7 @@
 fastapi>=0.100.0
 uvicorn>=0.22.0
 Pillow>=10.0.0
+tifffile>=2023.7.10
 pytest>=7.4.0
 pytest-qt>=4.2.0
 pytest-cov>=4.1.0
```

### `models/d04_ms5_plant.json`

```diff
--- a/models/d04_ms5_plant.json
+++ b/models/d04_ms5_plant.json
@@ -0,0 +1,7 @@
+{
+  "model_type": "yolox",
+  "classes": {"0": "plant"},
+  "input_profile": "ms5",
+  "confidence_threshold": 0.30,
+  "nms_iou_threshold": 0.45
+}
```

### `src/geo_weed_station/config/settings.py`

```diff
--- a/src/geo_weed_station/config/settings.py
+++ b/src/geo_weed_station/config/settings.py
@@ -1,6 +1,8 @@
 """Application settings module using Pydantic v2 BaseSettings."""
 
 from __future__ import annotations
+
+from typing import Literal
 
 import json
 from pathlib import Path
@@ -107,6 +109,7 @@
 class DetectorSettings(BaseModel):
     """ONNX neural network detector parameters."""
 
+    input_profile: Literal["rgb", "ms5"] = "rgb"
     model_path: Path = Path("models/weed_detection_yolo.onnx")
     classes_path: Path = Path("models/classes.json")
     confidence_threshold: float = 0.25
```

### `src/geo_weed_station/schemas/payload.py`

```diff
--- a/src/geo_weed_station/schemas/payload.py
+++ b/src/geo_weed_station/schemas/payload.py
@@ -18,6 +18,7 @@
     source_path: str | None = None
     image_size_px: tuple[int, int]  # (width, height)
     image: Any | None = Field(default=None, exclude=True)  # In-memory image buffer, excluded from serialization
+    detector_image: Any | None = Field(default=None, exclude=True)
     pose: CameraSpatialPose | None = None
     optics: SensorOptics | None = None
     detections: list[DetectedWeed] = Field(default_factory=list)
```

### `src/geo_weed_station/core/image_io.py`

```diff
--- a/src/geo_weed_station/core/image_io.py
+++ b/src/geo_weed_station/core/image_io.py
@@ -126,3 +126,62 @@
         raise
     except Exception as e:
         raise ImageReadError(f"Unexpected error decoding image {path.name}: {e}") from e
+
+
+def validate_ms5(image: np.ndarray) -> None:
+    """Validate normalized, registered MS5 data in R,G,B,RE,NIR order."""
+    if image.ndim != 3 or image.shape[2] != 5 or min(image.shape[:2]) < 1:
+        raise ImageReadError("MS5: нужны пять совмещённых каналов [R,G,B,RE,NIR]")
+    if image.dtype != np.float32 or not np.isfinite(image).all():
+        raise ImageReadError("MS5: ожидается конечный float32-массив")
+    if image.min() < 0 or image.max() > 1:
+        raise ImageReadError("MS5: значения float должны находиться в диапазоне [0,1]")
+
+
+def read_ms5_image(path: Path) -> np.ndarray:
+    """Read a registered five-band TIFF; integer scaling matches D04 inference.
+
+    Band order is a required input contract, not guessed from pixel values.
+    Accept interleaved YXS/YXC and planar SYX/CYX/QYX TIFF series.
+    """
+    import tifffile
+
+    path = Path(path)
+    if path.suffix.lower() not in {".tif", ".tiff"}:
+        raise ImageReadError("MS5: выберите пятиканальный TIFF [R,G,B,RE,NIR], не RGB-снимок")
+    try:
+        with tifffile.TiffFile(path) as tif:
+            series = tif.series[0]
+            axes = series.axes
+            if len(series.shape) != 3:
+                raise ImageReadError("MS5: TIFF должен содержать ровно пять полос одного кадра")
+            if axes in {"YXS", "YXC"} and series.shape[-1] == 5:
+                image = series.asarray()
+            elif axes in {"SYX", "CYX", "QYX", "IYX"} and series.shape[0] == 5:
+                image = series.asarray().transpose(1, 2, 0)
+            else:
+                raise ImageReadError(f"MS5: неподдерживаемые оси/форма TIFF: {axes} {series.shape}")
+        if image.dtype == np.uint8:
+            image = image.astype(np.float32) / 255.0
+        elif image.dtype == np.uint16:
+            image = image.astype(np.float32) / 65535.0
+        elif np.issubdtype(image.dtype, np.floating):
+            image = image.astype(np.float32)
+        else:
+            raise ImageReadError("MS5: поддерживаются TIFF uint8, uint16 и float [0,1]")
+        validate_ms5(image)
+        h, w = image.shape[:2]
+        if w > MAX_IMAGE_WIDTH or h > MAX_IMAGE_HEIGHT:
+            scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
+            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
+                               interpolation=cv2.INTER_AREA)
+        return np.ascontiguousarray(image)
+    except ImageReadError:
+        raise
+    except Exception as exc:
+        raise ImageReadError(f"Не удалось прочитать MS5 TIFF {path.name}: {exc}") from exc
+
+
+def ms5_rgb_preview(image: np.ndarray) -> np.ndarray:
+    """RGB uint8 on the same pixel grid; conversion matches reference D04 preview."""
+    return np.ascontiguousarray((image[..., :3] * 255.0).clip(0, 255).astype(np.uint8))
```

### `src/geo_weed_station/core/source_factory.py`

```diff
--- a/src/geo_weed_station/core/source_factory.py
+++ b/src/geo_weed_station/core/source_factory.py
@@ -11,7 +11,7 @@
 
 from geo_weed_station.core.annotation_io import AnnotationsMap, find_annotation_file, load_annotations_file
 from geo_weed_station.core.exceptions import ImageReadError
-from geo_weed_station.core.image_io import get_image_size, read_image
+from geo_weed_station.core.image_io import get_image_size, read_image, read_ms5_image, ms5_rgb_preview
 from geo_weed_station.core.xmp_parser import build_frame_payload, has_xmp
 from geo_weed_station.schemas.entities import DetectedWeed
 from geo_weed_station.schemas.onboard_api import SummaryDetections, summary_to_frame_payloads
@@ -51,17 +51,18 @@
     return FileEntry(path=str(path), size_bytes=size)
 
 
-def load_image_payload(path: Path, load_pixels: bool = True) -> FramePayload:
+def load_image_payload(path: Path, load_pixels: bool = True, input_profile: str = "rgb") -> FramePayload:
     """Decode an image and build a payload using only embedded XMP metadata.
     
     If load_pixels is False, only extracts dimensions and XMP metadata without decoding pixels.
     """
     path = Path(path)
     if load_pixels:
-        image = read_image(path)
+        ms5 = read_ms5_image(path) if input_profile == "ms5" else None
+        image = ms5_rgb_preview(ms5) if ms5 is not None else read_image(path)
         h, w = image.shape[:2]
         payload = build_frame_payload(path, (w, h))
-        return payload.model_copy(update={"image": image})
+        return payload.model_copy(update={"image": image, "detector_image": ms5})
     else:
         sz = get_image_size(path)
         return build_frame_payload(path, sz)
@@ -70,7 +71,8 @@
 class ImageSource:
     """Single image source."""
 
-    def __init__(self, path: Path) -> None:
+    def __init__(self, path: Path, input_profile: str = "rgb") -> None:
+        self.input_profile = input_profile
         self.path = Path(path)
         if not self.path.is_file():
             raise FileNotFoundError(f"Image not found: {self.path}")
@@ -79,13 +81,14 @@
         return [_entry_for(self.path)]
 
     def load(self, entry: FileEntry, load_pixels: bool = True) -> FramePayload:
-        return load_image_payload(Path(entry.path), load_pixels=load_pixels)
+        return load_image_payload(Path(entry.path), load_pixels=load_pixels, input_profile=self.input_profile)
 
 
 class FolderSource:
     """Recursive folder of images with optional preloaded annotations."""
 
-    def __init__(self, path: Path) -> None:
+    def __init__(self, path: Path, input_profile: str = "rgb") -> None:
+        self.input_profile = input_profile
         self.path = Path(path)
         if not self.path.is_dir():
             raise NotADirectoryError(f"Folder not found: {self.path}")
@@ -152,7 +155,7 @@
 
     def load(self, entry: FileEntry, load_pixels: bool = True) -> FramePayload:
         p = Path(entry.path)
-        payload = load_image_payload(p, load_pixels=load_pixels)
+        payload = load_image_payload(p, load_pixels=load_pixels, input_profile=self.input_profile)
         norm_str = str(p)
         dets = self._annotations_map.get(norm_str)
         if dets is None:
@@ -168,7 +171,8 @@
 class VideoSource:
     """Video file source; `load` returns the first frame (streaming is done by VideoWorker)."""
 
-    def __init__(self, path: Path) -> None:
+    def __init__(self, path: Path, input_profile: str = "rgb") -> None:
+        self.input_profile = input_profile
         self.path = Path(path)
         if not self.path.is_file():
             raise FileNotFoundError(f"Video not found: {self.path}")
@@ -204,7 +208,8 @@
 
     SUMMARY_NAME = "summary.json"
 
-    def __init__(self, path: Path) -> None:
+    def __init__(self, path: Path, input_profile: str = "rgb") -> None:
+        self.input_profile = input_profile
         self.path = Path(path)
         if not self.path.is_dir():
             raise NotADirectoryError(f"Run directory not found: {self.path}")
@@ -238,7 +243,7 @@
 
     def load(self, entry: FileEntry) -> FramePayload:
         path = Path(entry.path)
-        payload = load_image_payload(path)
+        payload = load_image_payload(path, input_profile=self.input_profile)
         idx = self._wp_index(path)
         if self._summary is not None and idx is not None:
             for fp in summary_to_frame_payloads(self._summary, payload.image_size_px):
@@ -251,13 +256,15 @@
     """Factory creating frame sources by kind."""
 
     @staticmethod
-    def create(kind: SourceKind, path: Path) -> FrameSource:
+    def create(kind: SourceKind, path: Path, input_profile: str = "rgb") -> FrameSource:
         if kind == "image":
-            return ImageSource(path)
+            return ImageSource(path, input_profile=input_profile)
         if kind == "folder":
-            return FolderSource(path)
+            return FolderSource(path, input_profile=input_profile)
         if kind == "video":
+            if input_profile == "ms5":
+                raise ImageReadError("MS5 не поддерживает RGB-видео")
             return VideoSource(path)
         if kind == "drone_run":
-            return DroneRunSource(path)
+            return DroneRunSource(path, input_profile=input_profile)
         raise ValueError(f"Unknown source kind: {kind!r}")
```

### `src/geo_weed_station/core/detector_base.py`

```diff
--- a/src/geo_weed_station/core/detector_base.py
+++ b/src/geo_weed_station/core/detector_base.py
@@ -17,11 +17,12 @@
         """Load detector model weights into execution session."""
 
     @abstractmethod
-    def detect(self, image: np.ndarray) -> list[DetectedWeed]:
+    def detect(self, image: np.ndarray, *, classification_image: np.ndarray | None = None) -> list[DetectedWeed]:
         """Perform object detection on an image array.
         
         Args:
-            image: RGB image numpy array (H, W, 3).
+            image: RGB uint8 (H,W,3), or normalized MS5 float32 (H,W,5).
+            classification_image: Optional aligned RGB uint8 for crop classification.
             
         Returns:
             List of DetectedWeed instances with normalized BBox [0..1] in original frame coordinates.
```

### `src/geo_weed_station/core/onnx_detector.py`

```diff
--- a/src/geo_weed_station/core/onnx_detector.py
+++ b/src/geo_weed_station/core/onnx_detector.py
@@ -438,6 +438,7 @@
     """Parsed model metadata including classes, architecture, and NMS thresholds."""
 
     classes: dict[int, str]
+    input_profile: str = "rgb"
     model_type: str | None = None
     confidence_threshold: float | None = None
     nms_iou_threshold: float | None = None
@@ -492,6 +493,7 @@
 
         return ModelMetadata(
             classes=classes,
+            input_profile="ms5" if data.get("input_profile") == "ms5" else "rgb",
             model_type=model_type,
             confidence_threshold=conf_thr,
             nms_iou_threshold=iou_thr,
@@ -611,6 +613,7 @@
                 classes_p = cand_cls
 
         meta = load_model_metadata(classes_p)
+        self.input_profile = "ms5" if meta.input_profile == "ms5" else settings.input_profile
         self._classes = meta.classes
         self._explicit_model_type = meta.model_type
         self.model_type: str = self._explicit_model_type or ModelType.AUTO.value
@@ -656,6 +659,7 @@
 
         if self.classifier_type == "c07":
             cands = [
+                get_models_dir() / "c07_rgb_geosaero" / "c07_rgb_geosaero_best.onnx",
                 get_models_dir() / "resnet50_dualhead_c07.onnx",
                 get_models_dir() / "best.onnx",
                 Path(__file__).resolve().parent.parent.parent.parent / "models" / "c07_rgb_geosaero" / "resnet50_dualhead_c07.onnx",
@@ -665,6 +669,7 @@
                     return c
         elif self.classifier_type == "c06":
             cands = [
+                get_models_dir() / "c06_rgb_species" / "c06_rgb_species_best.onnx",
                 get_models_dir() / "resnet50_c06.onnx",
                 Path(__file__).resolve().parent.parent.parent.parent / "models" / "c06_rgb_species" / "resnet50_c06.onnx",
             ]
@@ -679,9 +684,7 @@
             return
         cls_path = self._resolve_classifier_path()
         if cls_path is None or not cls_path.is_file():
-            logger.info("Classifier model for '%s' not found. Crop classification disabled.", self.classifier_type)
-            self._classifier_session = None
-            return
+            raise DetectionError(f"Модель классификации {self.classifier_type.upper()} не найдена")
 
         try:
             providers = self._get_providers()
@@ -693,7 +696,13 @@
                 sess_options=opts,
                 providers=providers,
             )
-            self._cls_input_name = self._classifier_session.get_inputs()[0].name
+            cls_input = self._classifier_session.get_inputs()[0]
+            if cls_input.shape != [1, 3, 224, 224] or cls_input.type != "tensor(float)":
+                raise DetectionError("C06/C07 ожидает вход float32 [1,3,224,224]")
+            expected = [[1, 11], [1, 5]] if self.classifier_type == "c07" else [[1, 24]]
+            if [o.shape for o in self._classifier_session.get_outputs()] != expected:
+                raise DetectionError("Выходы классификатора не соответствуют выбранной таксономии")
+            self._cls_input_name = cls_input.name
             self._cls_output_names = [o.name for o in self._classifier_session.get_outputs()]
             logger.info(
                 "Loaded ONNX classifier '%s' (%s) with %d outputs",
@@ -702,8 +711,8 @@
                 len(self._cls_output_names),
             )
         except Exception as e:
-            logger.warning("Failed to load classifier from %s: %s", cls_path, e)
             self._classifier_session = None
+            raise DetectionError(f"Не удалось загрузить классификатор {cls_path.name}: {e}") from e
 
     def set_classifier(self, classifier_type: str, classifier_path: Path | None = None) -> None:
         """Dynamically configure or switch classifier model."""
@@ -790,7 +799,14 @@
                 sess_options=session_options,
                 providers=providers,
             )
-            self._input_name = self._session.get_inputs()[0].name
+            inp = self._session.get_inputs()[0]
+            if self.input_profile == "ms5":
+                if inp.type != "tensor(float)" or inp.shape != [1, 5, 640, 640]:
+                    raise DetectionError(f"MS5 ожидает float32 [1,5,640,640], модель: {inp.type} {inp.shape}")
+                expected = [[1, 6, 80, 80], [1, 6, 40, 40], [1, 6, 20, 20]]
+                if [o.shape for o in self._session.get_outputs()] != expected:
+                    raise DetectionError("Неподдерживаемый экспорт MS5: нужны три карты YOLOX с логитами")
+            self._input_name = inp.name
             self._output_names = [out.name for out in self._session.get_outputs()]
             self.model_type = self._determine_model_type()
             logger.info(
@@ -805,7 +821,11 @@
 
         # Optional crop classifier initialization
         if self.classifier_type != "none":
-            self._load_classifier()
+            try:
+                self._load_classifier()
+            except DetectionError:
+                self.close()
+                raise
 
     def close(self) -> None:
         """Release ONNX InferenceSession and free GPU/system memory."""
@@ -820,8 +840,8 @@
         else:
             self.model_type = ModelType.AUTO.value
 
-    def detect(self, image: np.ndarray) -> list[DetectedWeed]:
-        """Run object detection on an RGB image array and return normalized weed detections."""
+    def detect(self, image: np.ndarray, *, classification_image: np.ndarray | None = None) -> list[DetectedWeed]:
+        """Detect on RGB/MS5; classify aligned RGB crops and return normalized boxes."""
         if not self.is_loaded:
             self.load()
 
@@ -832,13 +852,31 @@
         if orig_h <= 0 or orig_w <= 0:
             return []
 
-        # Letterbox preprocessing
-        lb = letterbox(image, size=self.settings.input_size)
-        pad_x, pad_y = lb.pad
-        scale = lb.scale
+        if self.input_profile == "ms5":
+            from geo_weed_station.core.image_io import validate_ms5, ms5_rgb_preview
+            validate_ms5(image)
+            # D04 reference inference stretches each band to 640x640 (no letterbox).
+            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
+            tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
+            scale_x, scale_y = 640.0 / orig_w, 640.0 / orig_h
+            pad_x = pad_y = 0.0
+            if classification_image is None:
+                classification_image = ms5_rgb_preview(image)
+        else:
+            if image.ndim != 3 or image.shape[2] != 3:
+                raise DetectionError("RGB-модель ожидает изображение с тремя каналами")
+            lb = letterbox(image, size=self.settings.input_size)
+            tensor = lb.tensor
+            pad_x, pad_y = lb.pad
+            scale_x = scale_y = lb.scale
+            if classification_image is None:
+                classification_image = image
+        if (classification_image.shape != (orig_h, orig_w, 3)
+                or classification_image.dtype != np.uint8):
+            raise DetectionError("Классификатор ожидает совмещённый RGB uint8 размером с исходный кадр")
 
         try:
-            outputs = self._session.run(None, {self._input_name: lb.tensor})
+            outputs = self._session.run(None, {self._input_name: tensor})
         except Exception as e:
             raise DetectionError(f"Inference execution failed: {e}") from e
 
@@ -849,7 +887,7 @@
         if self.model_type == ModelType.YOLOX.value:
             boxes_arr, scores_arr, class_ids = _decode_yolox(
                 outputs,
-                input_size=self.settings.input_size,
+                input_size=(640, 640) if self.input_profile == "ms5" else self.settings.input_size,
                 conf_threshold=self.confidence_threshold,
             )
         else:
@@ -876,10 +914,10 @@
             class_name = self._classes.get(c_id, f"class_{c_id}")
 
             # Inverse letterbox transformation back to original image space
-            orig_x1 = (bx1 - pad_x) / scale
-            orig_y1 = (by1 - pad_y) / scale
-            orig_x2 = (bx2 - pad_x) / scale
-            orig_y2 = (by2 - pad_y) / scale
+            orig_x1 = (bx1 - pad_x) / scale_x
+            orig_y1 = (by1 - pad_y) / scale_y
+            orig_x2 = (bx2 - pad_x) / scale_x
+            orig_y2 = (by2 - pad_y) / scale_y
 
             # Clip to [0, orig_w] and [0, orig_h]
             orig_x1 = max(0.0, min(float(orig_w), orig_x1))
@@ -900,7 +938,7 @@
                 cy2 = min(orig_h, int(round(orig_y2)) + pad_h)
 
                 if (cx2 - cx1) >= 4 and (cy2 - cy1) >= 4:
-                    crop_rgb = image[cy1:cy2, cx1:cx2]
+                    crop_rgb = classification_image[cy1:cy2, cx1:cx2]
                     target_cls_w, target_cls_h = self.settings.classifier_input_size
                     crop_res = cv2.resize(crop_rgb, (target_cls_w, target_cls_h), interpolation=cv2.INTER_LINEAR)
                     crop_norm = (crop_res.astype(np.float32) / 255.0 - mean_rgb) / std_rgb
@@ -930,7 +968,7 @@
                             class_prob = float(p[sp_id])
                             c_id = sp_id
                     except Exception as exc:
-                        logger.warning("Error running ONNX crop classification: %s", exc)
+                        raise DetectionError(f"Ошибка классификации RGB-кропа: {exc}") from exc
 
             # Normalize to [0.0..1.0]
             norm_x = min(1.0, max(0.0, orig_x1 / orig_w))
```

### `src/geo_weed_station/workers/batch_image_worker.py`

```diff
--- a/src/geo_weed_station/workers/batch_image_worker.py
+++ b/src/geo_weed_station/workers/batch_image_worker.py
@@ -75,7 +75,12 @@
                     count: int | None = len(payload.detections) if payload.detections else None
                     if run_inference:
                         detector = self._ensure_detector()
-                        detections = detector.detect(payload.image)
+                        if payload.detector_image is None:
+                            detections = detector.detect(payload.image)
+                        else:
+                            detections = detector.detect(
+                                payload.detector_image, classification_image=payload.image
+                            )
                         payload = payload.model_copy(update={"detections": detections})
                         count = len(detections)
                     status = (
@@ -84,9 +89,11 @@
                         else ProcessingStatus.DONE
                     )
                     # Frames cross the thread boundary as copies only
-                    safe = payload.model_copy(
-                        update={"image": None if payload.image is None else payload.image.copy()}
-                    )
+                    rgb = None if payload.image is None else payload.image.copy()
+                    if rgb is not None:
+                        rgb.setflags(write=False)
+                    # The full MS5 cube is only needed in this worker, not in the GUI.
+                    safe = payload.model_copy(update={"image": rgb, "detector_image": None})
                     self._set_status(i, status, count)
                     self.frame_processed.emit(safe)
                     ok += 1
```

### `src/geo_weed_station/workers/ms5_preview_worker.py`

```diff
--- a/src/geo_weed_station/workers/ms5_preview_worker.py
+++ b/src/geo_weed_station/workers/ms5_preview_worker.py
@@ -0,0 +1,30 @@
+"""MS5 preview decoding outside the GUI thread."""
+from pathlib import Path
+
+from PySide6.QtCore import Signal, Slot
+
+from geo_weed_station.core.image_io import read_ms5_image, ms5_rgb_preview
+from geo_weed_station.workers.base_worker import ManagedWorker
+
+
+class Ms5PreviewWorker(ManagedWorker):
+    ready = Signal(int, str, object)
+
+    def __init__(self, path: Path, generation: int) -> None:
+        super().__init__()
+        self.path = path
+        self.generation = generation
+
+    @Slot()
+    def start_work(self) -> None:
+        try:
+            if not self.is_stop_requested:
+                image = ms5_rgb_preview(read_ms5_image(self.path))
+                image.setflags(write=False)
+                if not self.is_stop_requested:
+                    self.ready.emit(self.generation, str(self.path), image)
+        except Exception as exc:
+            if not self.is_stop_requested:
+                self.failed.emit(str(exc))
+        finally:
+            self.finished.emit()
```

### `src/geo_weed_station/workers/thumbnail_loader.py`

```diff
--- a/src/geo_weed_station/workers/thumbnail_loader.py
+++ b/src/geo_weed_station/workers/thumbnail_loader.py
@@ -0,0 +1,32 @@
+"""Load mosaic RGB thumbnails in the thread pool, deliver in the GUI thread."""
+from PySide6.QtCore import QRunnable
+
+from geo_weed_station.core.image_io import read_image, read_ms5_image, ms5_rgb_preview
+from geo_weed_station.core.map_engine.mosaic import make_thumbnail
+
+
+class ThumbnailJob(QRunnable):
+    def __init__(self, loader):
+        super().__init__()
+        self.loader = loader
+
+    def run(self):
+        if self.loader.callback() is None:
+            return
+        try:
+            path = self.loader.path
+            is_ms5 = False
+            if path.suffix.lower() in {'.tif', '.tiff'}:
+                import tifffile
+                with tifffile.TiffFile(path) as tif:
+                    series = tif.series[0]
+                    is_ms5 = (len(series.shape) == 3 and (
+                        (series.axes in {'YXS', 'YXC'} and series.shape[-1] == 5)
+                        or (series.axes in {'SYX', 'CYX', 'QYX', 'IYX'} and series.shape[0] == 5)))
+            rgb = ms5_rgb_preview(read_ms5_image(path)) if is_ms5 else read_image(path)
+            thumbnail = make_thumbnail(rgb)
+            thumbnail.setflags(write=False)
+            self.loader.result.emit(thumbnail)
+        except Exception:
+            # A missing/corrupt raster leaves the footprint outline visible.
+            self.loader.result.emit(None)
```

### `src/geo_weed_station/controllers/thumbnail_controller.py`

```diff
--- a/src/geo_weed_station/controllers/thumbnail_controller.py
+++ b/src/geo_weed_station/controllers/thumbnail_controller.py
@@ -0,0 +1,27 @@
+"""Dispatch background thumbnails to surviving graphics items in the GUI thread."""
+from pathlib import Path
+import weakref
+
+from PySide6.QtCore import QObject, QThreadPool, Qt, Signal, Slot
+from shiboken6 import isValid
+
+from geo_weed_station.workers.thumbnail_loader import ThumbnailJob
+
+
+class ThumbnailController(QObject):
+    result = Signal(object)
+
+    def __init__(self, path, callback):
+        super().__init__()
+        self.path = Path(path)
+        self.callback = weakref.WeakMethod(callback)
+        self.result.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
+
+    def start(self):
+        QThreadPool.globalInstance().start(ThumbnailJob(self))
+
+    @Slot(object)
+    def _deliver(self, thumbnail):
+        callback = self.callback()
+        if thumbnail is not None and callback is not None and isValid(callback.__self__):
+            callback(thumbnail)
```

### `src/geo_weed_station/ui/widgets/detection_filter_panel.py`

```diff
--- a/src/geo_weed_station/ui/widgets/detection_filter_panel.py
+++ b/src/geo_weed_station/ui/widgets/detection_filter_panel.py
@@ -85,8 +85,8 @@
 
         self._combo_classifier = QComboBox(self)
         self._combo_classifier.addItem("Отключена", userData="none")
-        self._combo_classifier.addItem("C07 (7 видов + фазы)", userData="c07")
-        self._combo_classifier.addItem("C06 (6 видов)", userData="c06")
+        self._combo_classifier.addItem("C07 (категории + фазы)", userData="c07")
+        self._combo_classifier.addItem("C06 (виды растений)", userData="c06")
         self._combo_classifier.currentIndexChanged.connect(self._on_combo_classifier_changed)
         cls_layout.addWidget(self._combo_classifier)
 
@@ -189,19 +189,30 @@
 
         for i, m in enumerate(models):
             p = Path(m)
-            self._combo_model.addItem(p.name, userData=str(p))
+            self._combo_model.addItem(
+                "D04 MS5 — 5 каналов" if p.stem == "d04_ms5_plant" else p.name,
+                userData=str(p),
+            )
             if active_resolved is not None and p.exists() and p.resolve() == active_resolved:
                 active_idx = i
 
         if active_idx == -1 and active is not None and Path(active).is_file():
             p = Path(active)
-            self._combo_model.addItem(p.name, userData=str(p))
+            self._combo_model.addItem(
+                "D04 MS5 — 5 каналов" if p.stem == "d04_ms5_plant" else p.name,
+                userData=str(p),
+            )
             active_idx = self._combo_model.count() - 1
 
         if active_idx >= 0:
             self._combo_model.setCurrentIndex(active_idx)
 
         self._block_signals = False
+
+    def set_model_controls_enabled(self, enabled: bool) -> None:
+        self._combo_model.setEnabled(enabled)
+        self._btn_browse_model.setEnabled(enabled)
+        self._combo_classifier.setEnabled(enabled)
 
     def selected_model(self) -> Path | None:
         """Return currently selected ONNX model path or None."""
```

### `src/geo_weed_station/controllers/offline_controller.py`

```diff
--- a/src/geo_weed_station/controllers/offline_controller.py
+++ b/src/geo_weed_station/controllers/offline_controller.py
@@ -9,7 +9,7 @@
 from PySide6.QtCore import QObject, Qt, Signal, Slot
 from PySide6.QtWidgets import QApplication, QFileDialog, QProgressDialog
 
-from geo_weed_station.config.settings import AppSettings
+from geo_weed_station.config.settings import AppSettings, DetectorSettings
 from geo_weed_station.controllers.map_controller import MapController
 from geo_weed_station.core.annotation_io import (
     AnnotationsMap,
@@ -22,6 +22,7 @@
 from geo_weed_station.core.onnx_detector import (
     OnnxDetector,
     get_models_dir,
+    load_model_metadata,
     resolve_classes_path,
     scan_available_models,
 )
@@ -56,6 +57,7 @@
 from geo_weed_station.workers.base_worker import start_in_thread
 from geo_weed_station.workers.batch_image_worker import BatchImageWorker
 from geo_weed_station.workers.video_worker import VideoWorker
+from geo_weed_station.workers.ms5_preview_worker import Ms5PreviewWorker
 
 logger = logging.getLogger(__name__)
 
@@ -93,9 +95,20 @@
         # Active worker references
         self._batch_worker: BatchImageWorker | None = None
         self._video_worker: VideoWorker | None = None
+        self._batch_errors: list[str] = []
+        self._preview_generation = 0
+        self._preview_worker = None
         self._current_source_folder: Path | None = None
         self._detections_by_path: dict[str, list[DetectedWeed]] = {}
 
+        saved = self._window._qsettings.value("offline/detector") if self._window._qsettings else None
+        if saved and not self._has_custom_detector_factory:
+            try:
+                restored = DetectorSettings.model_validate_json(saved)
+                if restored.model_path.is_file():
+                    self._settings = self._settings.model_copy(update={"detector": restored})
+            except (ValueError, TypeError):
+                logger.warning("Ignoring invalid saved detector selection")
         self._embed_widgets_in_ui()
         self._wire_signals()
         self._init_models_in_ui()
@@ -128,7 +141,9 @@
 
         if resolved_model.is_file():
             updated_det = self._settings.detector.model_copy(
-                update={"model_path": resolved_model, "classes_path": classes_p}
+                update={"model_path": resolved_model, "classes_path": classes_p,
+                        "input_profile": ("rgb" if self._has_custom_detector_factory else
+                                          load_model_metadata(classes_p).input_profile)}
             )
             self._settings = self._settings.model_copy(update={"detector": updated_det})
             if not self._has_custom_detector_factory:
@@ -136,9 +151,13 @@
 
     def apply_settings(self, settings: AppSettings) -> None:
         """Dynamically apply updated application settings and refresh model configuration."""
+        if self._batch_worker is not None:
+            self._window.status_bar_panel.set_message("Сначала завершите обработку")
+            return
         self._settings = settings
         self._detector_factory = lambda: OnnxDetector(self._settings.detector)
         self._init_models_in_ui()
+        self._save_detector_selection()
 
     def _embed_widgets_in_ui(self) -> None:
         # Install content widgets into central tabs
@@ -182,15 +201,29 @@
 
     def _on_model_changed(self, model_path: Path) -> None:
         """Handle model switch: resolve classes, reconfigure factory and update UI filter checkboxes."""
-        classes_path = resolve_classes_path(model_path, self._settings.detector.classes_path)
+        if self._batch_worker is not None:
+            self._init_models_in_ui()
+            return
+        classes_path = resolve_classes_path(model_path, get_models_dir() / "classes.json")
+        profile = load_model_metadata(classes_path).input_profile
+        if profile == "ms5" and self._video_worker is not None:
+            self._video_worker.request_stop()
+            self._video_worker = None
+            self.video_player.clear()
+        self._has_custom_detector_factory = False
+        self._clear_model_results()
         logger.info("Switching detection model to %s with classes %s", model_path.name, classes_path.name)
 
         updated_det = self._settings.detector.model_copy(
-            update={"model_path": model_path, "classes_path": classes_path}
+            update={"model_path": model_path, "classes_path": classes_path,
+                    "input_profile": profile, "classifier_type": "none",
+                    "classifier_path": None}
         )
         self._settings = self._settings.model_copy(update={"detector": updated_det})
         self._detector_factory = lambda: OnnxDetector(self._settings.detector)
 
+        self.filter_panel.set_classifier_type("none")
+        self._save_detector_selection()
         if classes_path.is_file():
             self.filter_panel.load_classes_from_file(classes_path)
 
@@ -203,9 +236,13 @@
 
     def _on_classifier_changed(self, cls_type: str) -> None:
         """Handle crop classifier switch: update settings and detector factory."""
+        if self._batch_worker is not None:
+            self.filter_panel.set_classifier_type(self._settings.detector.classifier_type)
+            return
+        self._clear_model_results()
         logger.info("Switching crop classifier to %s", cls_type)
         updated_det = self._settings.detector.model_copy(
-            update={"classifier_type": cls_type}
+            update={"classifier_type": cls_type, "classifier_path": None}
         )
         self._settings = self._settings.model_copy(update={"detector": updated_det})
         self._detector_factory = lambda: OnnxDetector(self._settings.detector)
@@ -213,10 +250,11 @@
         if self._video_worker is not None:
             self._video_worker.set_detector_factory(self._detector_factory)
 
+        self._save_detector_selection()
         labels = {
             "none": "Отключена",
-            "c07": "C07 (7 видов + фазы)",
-            "c06": "C06 (6 видов)",
+            "c07": "C07 (категории + фазы)",
+            "c06": "C06 (виды растений)",
         }
         lbl = labels.get(cls_type.lower(), cls_type)
         self._window.status_bar_panel.set_message(f"Классификация видов: {lbl}")
@@ -227,6 +265,9 @@
     @Slot()
     def open_image(self) -> None:
         """Open a single image file, populate file list panel and preview without auto-inference."""
+        if self._batch_worker is not None:
+            self._window.status_bar_panel.set_message("Сначала завершите обработку")
+            return
         exts = " ".join(f"*{e}" for e in IMAGE_EXTENSIONS)
         path, _ = QFileDialog.getOpenFileName(
             self._window, "Открыть изображение", str(self._settings.workdir), f"Images ({exts})"
@@ -330,6 +371,9 @@
     @Slot()
     def open_folder(self) -> None:
         """Open directory of images and populate file list panel."""
+        if self._batch_worker is not None:
+            self._window.status_bar_panel.set_message("Сначала завершите обработку")
+            return
         folder = QFileDialog.getExistingDirectory(
             self._window, "Выбрать каталог снимков", str(self._settings.workdir)
         )
@@ -498,9 +542,33 @@
             logger.exception("Failed to save annotations: %s", exc)
             self._window.status_bar_panel.set_message(f"Ошибка сохранения разметки: {exc}")
 
+    def _save_detector_selection(self) -> None:
+        if self._window._qsettings is not None:
+            self._window._qsettings.setValue("offline/detector", self._settings.detector.model_dump_json())
+
+    def _clear_model_results(self) -> None:
+        self._preview_generation += 1
+        if self._preview_worker is not None:
+            self._preview_worker.request_stop()
+        self._detections_by_path.clear()
+        self.image_viewer.clear()
+        self._map_ctrl.reset()
+        self.map_info_panel.clear_selection()
+        for row, entry in enumerate(self.file_list_panel.model.entries):
+            entry.detections_count = None
+            self.file_list_panel.update_status(row, ProcessingStatus.PENDING)
+        self.map_canvas.set_mosaic_footprints([])
+        self.map_canvas.set_prescription_zones([])
+        self.map_canvas.clear_route()
+
     @Slot()
     def open_video(self) -> None:
         """Open video file, initialize background VideoWorker and wire playback controls."""
+        if self._batch_worker is not None:
+            return
+        if self._settings.detector.input_profile == "ms5":
+            self._window.status_bar_panel.set_message("D04 MS5 требует пятиканальный TIFF; RGB-видео не поддерживается")
+            return
         exts = " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)
         path, _ = QFileDialog.getOpenFileName(
             self._window, "Открыть видеофайл", str(self._settings.workdir), f"Videos ({exts})"
@@ -576,6 +644,8 @@
     @Slot()
     def start_batch(self) -> None:
         """Launch background BatchImageWorker processing across loaded folder entries."""
+        if self._batch_worker is not None:
+            return
         entries = self.file_list_panel.model.entries
         if not entries:
             self._window.status_bar_panel.set_message("Пакетный список файлов пуст")
@@ -586,22 +656,26 @@
         self.map_info_panel.clear_selection()
 
         if len(entries) == 1 and Path(entries[0].path).is_file():
-            source = ImageSource(Path(entries[0].path))
+            source = ImageSource(Path(entries[0].path), input_profile=self._settings.detector.input_profile)
         else:
             source_folder = self._current_source_folder or Path(entries[0].path).parent
-            source = FolderSource(source_folder)
+            source = FolderSource(source_folder, input_profile=self._settings.detector.input_profile)
 
         worker = BatchImageWorker(
             source=source,
             detector_factory=self._detector_factory,
         )
+        self._batch_errors.clear()
         self._batch_worker = worker
+        self.filter_panel.set_model_controls_enabled(False)
+        self.file_list_panel.btn_clear.setEnabled(False)
 
         # Connect lifecycle signals
         worker.progress.connect(self._on_batch_progress)
         worker.frame_processed.connect(self._on_batch_frame_processed)
         worker.entry_status_changed.connect(self.file_list_panel.update_status)
         worker.batch_finished.connect(self._on_batch_finished)
+        worker.failed.connect(self._on_batch_error)
 
         # Enable cancel button in status bar
         self._window.status_bar_panel.set_cancel_enabled(True)
@@ -658,6 +732,18 @@
             self._load_entry_to_viewer(entry)
 
     def _load_entry_to_viewer(self, entry: FileEntry) -> None:
+        self._preview_generation += 1
+        if self._preview_worker is not None:
+            self._preview_worker.request_stop()
+        if self._settings.detector.input_profile == "ms5":
+            self.image_viewer.clear()
+            worker = Ms5PreviewWorker(Path(entry.path), self._preview_generation)
+            self._preview_worker = worker
+            worker.ready.connect(self._on_ms5_preview)
+            worker.failed.connect(self._window.status_bar_panel.set_message)
+            thread = start_in_thread(worker, "ms5_preview")
+            self._window.register_thread(thread, worker)
+            return
         try:
             path = Path(entry.path)
             if path.is_file():
@@ -670,7 +756,17 @@
         except Exception as exc:
             logger.warning("Could not preview file %s: %s", entry.path, exc)
 
+    @Slot(int, str, object)
+    def _on_ms5_preview(self, generation: int, path: str, image) -> None:
+        if generation != self._preview_generation:
+            return
+        self.image_viewer.set_image(image)
+        self.image_viewer.set_detections(self._detections_by_path.get(path, []))
+
     def _on_file_list_cleared(self) -> None:
+        self._preview_generation += 1
+        if self._preview_worker is not None:
+            self._preview_worker.request_stop()
         self._detections_by_path.clear()
         self.image_viewer.clear()
         self._map_ctrl.reset()
@@ -704,6 +800,8 @@
             # If currently selected item is this file, update detection overlay immediately
             selected = self.file_list_panel.selected_entry()
             if selected is not None and selected.path in (payload.source_path, norm_path):
+                if payload.image is not None:
+                    self.image_viewer.set_image(payload.image)
                 self.image_viewer.set_detections(detections)
 
         # Merge classes into filter panel
@@ -727,6 +825,11 @@
         else:
             self._map_ctrl.map_locked_changed.emit(True, MAP_LOCKED_TEXT)
 
+    @Slot(str)
+    def _on_batch_error(self, message: str) -> None:
+        self._batch_errors.append(message)
+        self._window.status_bar_panel.set_message(message)
+
     def _on_batch_finished(self, succeeded: int, failed: int) -> None:
         try:
             self.batch_start_requested.disconnect()
@@ -736,8 +839,11 @@
         self._window.status_bar_panel.set_progress(100)
         self._window.status_bar_panel.set_message(
             f"Обработка завершена: успешно {succeeded}, ошибок {failed}"
+            + (f". {self._batch_errors[-1]}" if self._batch_errors else "")
         )
         self._batch_worker = None
+        self.filter_panel.set_model_controls_enabled(True)
+        self.file_list_panel.btn_clear.setEnabled(True)
 
         # Build final aggregated layers: deduplicated points, KDE heatmap, prescription zones, mosaic, route
         self._on_map_updated()
```

### `src/geo_weed_station/ui/dialogs/settings_dialog.py`

```diff
--- a/src/geo_weed_station/ui/dialogs/settings_dialog.py
+++ b/src/geo_weed_station/ui/dialogs/settings_dialog.py
@@ -296,6 +296,10 @@
                 reconnect_backoff_max_s=self.spin_backoff_max.value(),
             )
             det = DetectorSettings(
+                input_profile=self._initial_settings.detector.input_profile,
+                classifier_type=self._initial_settings.detector.classifier_type,
+                classifier_path=self._initial_settings.detector.classifier_path,
+                classifier_input_size=self._initial_settings.detector.classifier_input_size,
                 model_path=Path(self.edit_model_path.text()),
                 classes_path=Path(self.edit_classes_path.text()),
                 confidence_threshold=self.spin_conf_thresh.value(),
```

### `src/geo_weed_station/controllers/app_controller.py`

```diff
--- a/src/geo_weed_station/controllers/app_controller.py
+++ b/src/geo_weed_station/controllers/app_controller.py
@@ -106,6 +106,10 @@
     # Settings Handler
     # -------------------------------------------------------------------------
     def _open_settings_dialog(self) -> None:
+        if self.offline_controller._batch_worker is not None:
+            self.window.status_bar_panel.set_message("Сначала завершите обработку")
+            return
+        self.settings = self.offline_controller._settings
         dialog = SettingsDialog(self.settings, parent=self.window)
         if dialog.exec() == QDialog.DialogCode.Accepted:
             self.settings = dialog.result_settings()
```

### `src/geo_weed_station/ui/widgets/map_canvas.py`

```diff
--- a/src/geo_weed_station/ui/widgets/map_canvas.py
+++ b/src/geo_weed_station/ui/widgets/map_canvas.py
@@ -5,9 +5,7 @@
 import logging
 import math
 from collections.abc import Sequence
-from pathlib import Path
-
-import cv2
+
 import numpy as np
 from PySide6.QtCore import QPointF, QRectF, Qt, Signal, Slot
 from PySide6.QtGui import (
@@ -130,25 +128,22 @@
                 th_h, th_w = th.shape[:2]
                 qimg = QImage(th.data, th_w, th_h, th_w * 3, QImage.Format.Format_RGB888).copy()
                 pm = QPixmap.fromImage(qimg)
-            elif footprint.source_path and Path(footprint.source_path).is_file():
-                try:
-                    img = cv2.imread(str(footprint.source_path))
-                    if img is not None:
-                        h, w = img.shape[:2]
-                        scale = 400.0 / float(max(h, w))
-                        thumb = cv2.resize(
-                            img,
-                            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
-                            interpolation=cv2.INTER_AREA,
-                        )
-                        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
-                        th_h, th_w = thumb_rgb.shape[:2]
-                        qimg = QImage(thumb_rgb.data, th_w, th_h, th_w * 3, QImage.Format.Format_RGB888).copy()
-                        pm = QPixmap.fromImage(qimg)
-                except Exception:
-                    pass
 
         if pm is not None and not pm.isNull():
+            self._set_thumbnail_pixmap(pm)
+        elif footprint.source_path:
+            from geo_weed_station.controllers.thumbnail_controller import ThumbnailController
+            self._thumbnail_loader = ThumbnailController(footprint.source_path, self._on_thumbnail)
+            self._thumbnail_loader.start()
+
+    def _on_thumbnail(self, thumbnail) -> None:
+        h, w = thumbnail.shape[:2]
+        qimg = QImage(thumbnail.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
+        self._set_thumbnail_pixmap(QPixmap.fromImage(qimg))
+
+    def _set_thumbnail_pixmap(self, pm: QPixmap) -> None:
+        footprint = self.footprint
+        if not pm.isNull():
             pix_item = QGraphicsPixmapItem(pm, self)
             pix_item.setOpacity(0.85)
             pw = pm.width()
```

### `tests/test_ms5_integration.py`

```diff
--- a/tests/test_ms5_integration.py
+++ b/tests/test_ms5_integration.py
@@ -0,0 +1,235 @@
+"""MS5 data contracts, geometry, classification and real ONNX integration."""
+from pathlib import Path
+from types import SimpleNamespace
+
+import cv2
+import numpy as np
+import pytest
+import tifffile
+
+from geo_weed_station.config.settings import DetectorSettings
+from geo_weed_station.core.exceptions import ImageReadError, DetectionError
+from geo_weed_station.core.image_io import read_ms5_image, ms5_rgb_preview
+from geo_weed_station.core.onnx_detector import OnnxDetector, _decode_yolox, nms
+from geo_weed_station.core.source_factory import ImageSource, SourceFactory
+from geo_weed_station.workers.batch_image_worker import BatchImageWorker
+
+ROOT = Path(__file__).resolve().parents[1]
+
+
+def write_ms5(path, array, planar=False, xmp=None):
+    extra = [] if xmp is None else [(700, 'B', len(xmp.encode()), xmp.encode(), False)]
+    tifffile.imwrite(path, array.transpose(2, 0, 1) if planar else array,
+                     photometric='minisblack', planarconfig='separate' if planar else 'contig',
+                     metadata={'axes': 'SYX' if planar else 'YXS'}, extratags=extra)
+    return path
+
+
+@pytest.mark.parametrize('dtype,maximum', [(np.uint8, 255), (np.uint16, 65535), (np.float32, 1)])
+@pytest.mark.parametrize('planar', [False, True])
+def test_band_order_and_normalization(tmp_path, dtype, maximum, planar):
+    image = np.broadcast_to(np.array([0, .2, .4, .7, 1]) * maximum, (12, 20, 5)).astype(dtype)
+    p = write_ms5(tmp_path/'ms.tif', image, planar)
+    actual = read_ms5_image(p)
+    np.testing.assert_allclose(actual, image.astype(np.float32)/maximum, atol=1e-7)
+    assert actual.flags.c_contiguous
+    np.testing.assert_array_equal(ms5_rgb_preview(actual), (actual[..., :3]*255).astype(np.uint8))
+
+
+@pytest.mark.parametrize('bad', ['rgb', 'nan', 'range', 'signed'])
+def test_invalid_ms5_rejected(tmp_path, bad):
+    image = np.zeros((12, 20, 5), np.float32)
+    if bad == 'rgb':
+        image = image[..., :3]
+    elif bad == 'nan':
+        image[0, 0, 4] = np.nan
+    elif bad == 'range':
+        image[0, 0, 3] = 2
+    else:
+        image = image.astype(np.int16)
+    p = write_ms5(tmp_path/'bad.tif', image)
+    with pytest.raises(ImageReadError):
+        read_ms5_image(p)
+
+
+def test_rgb_source_and_video_rejected(tmp_path):
+    p = tmp_path/'rgb.jpg'
+    cv2.imwrite(str(p), np.zeros((12, 20, 3), np.uint8))
+    with pytest.raises(ImageReadError, match='TIFF'):
+        ImageSource(p, input_profile='ms5').load(ImageSource(p).entries()[0])
+    with pytest.raises(ImageReadError, match='видео'):
+        SourceFactory.create('video', p, input_profile='ms5')
+
+
+def synthetic_detector(tmp_path, classification='c06'):
+    meta = tmp_path/'ms5.json'
+    meta.write_text('{"model_type":"yolox","input_profile":"ms5","classes":{"0":"plant"}}')
+    det = OnnxDetector(DetectorSettings(classes_path=meta, classifier_type=classification))
+    outputs = [np.full((1, 6, s, s), -30, np.float32) for s in (80, 40, 20)]
+    # Box [160,160,480,480] in resized 640x640 space.
+    outputs[0][0, :, 40, 40] = [0, 0, np.log(40), np.log(40), 20, 20]
+    calls = {}
+    def run(_, feed):
+        calls['detector'] = next(iter(feed.values()))
+        return outputs
+    det._session = SimpleNamespace(run=run)
+    det._input_name = 'ms_images'
+    def classify(_, feed):
+        calls['classifier'] = next(iter(feed.values()))
+        logits = np.full((1, 24), -10, np.float32)
+        logits[0, 2] = 10
+        return [logits]
+    if classification != 'none':
+        det._classifier_session = SimpleNamespace(run=classify)
+        det._cls_input_name = 'images'
+    return det, calls
+
+
+def test_resize_inverse_geometry_and_rgb_crop(tmp_path):
+    det, calls = synthetic_detector(tmp_path)
+    image = np.broadcast_to(np.array([.1,.2,.3,.8,.9], np.float32), (100, 200, 5)).copy()
+    rgb = ms5_rgb_preview(image)
+    result = det.detect(image, classification_image=rgb)
+    assert len(result) == 1
+    assert result[0].bbox.model_dump() == pytest.approx(dict(x=.25,y=.25,w=.5,h=.5))
+    assert result[0].class_id == 2
+    assert result[0].class_prob == 1
+    assert calls['detector'].shape == (1,5,640,640)
+    np.testing.assert_allclose(calls['detector'][0,:,0,0], image[0,0])
+    assert calls['classifier'].shape == (1,3,224,224)
+    expected = (rgb[0,0]/255.0-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
+    np.testing.assert_allclose(calls['classifier'][0,:,0,0], expected, atol=1e-6)
+    with pytest.raises(DetectionError, match='совмещённый'):
+        det.detect(image, classification_image=rgb[:10])
+
+
+def test_batch_preserves_xmp_and_emits_rgb_only(tmp_path):
+    from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO
+    path = write_ms5(tmp_path/'frame.tif', np.full((100,200,5), .5, np.float32),
+                     xmp=SAMPLE_XMP_LOCAL_ARUCO)
+    source = ImageSource(path, input_profile='ms5')
+    det, _ = synthetic_detector(tmp_path, 'none')
+    det.load = lambda: None
+    worker = BatchImageWorker(source, detector_factory=lambda: det)
+    received, finished = [], []
+    worker.frame_processed.connect(received.append)
+    worker.batch_finished.connect(lambda ok, failed: finished.append((ok,failed)))
+    worker.process(source.entries(), True)
+    assert finished == [(1,0)]
+    payload = received[0]
+    assert payload.has_georef()
+    assert payload.detector_image is None
+    assert payload.image.shape == (100,200,3)
+    assert not payload.image.flags.writeable
+    assert payload.detections[0].class_name == 'plant'
+    from geo_weed_station.core.map_engine.weed_map import MapEngine
+    from geo_weed_station.core.map_engine.projector import NadirProjector
+    from geo_weed_station.config.settings import MapSettings
+    engine = MapEngine(projector=NadirProjector(), settings=MapSettings())
+    engine.ingest_frame(payload)
+    assert engine.statistics().total_weeds == 1
+    assert engine.weed_map.footprints[0].thumbnail.shape[2] == 3
+    from geo_weed_station.core.annotation_io import save_annotations_to_folder, load_annotations_file
+    saved = save_annotations_to_folder(tmp_path, {str(path): payload.detections}, source.entries())
+    assert load_annotations_file(saved, [path])[str(path)][0].class_name == 'plant'
+
+
+def test_cancel_batch_does_not_decode(tmp_path):
+    source = SimpleNamespace(load=lambda entry: pytest.fail('cancelled load'))
+    worker = BatchImageWorker(source)
+    worker.cancel()
+    worker.process([SimpleNamespace(path='unused')], True)
+
+
+@pytest.mark.parametrize('kind', ['c06', 'c07'])
+def test_real_models_two_stage(tmp_path, kind):
+    model = ROOT/'models/d04_ms5_plant.onnx'
+    if not model.exists():
+        pytest.skip('Local D04 weights are not installed')
+    det = OnnxDetector(DetectorSettings(model_path=model,
+        classes_path=model.with_suffix('.json'), classifier_type=kind), providers=['CPUExecutionProvider'])
+    try:
+        det.load()
+        assert det._classifier_session is not None
+        image = np.full((80,120,5), .5, np.float32)
+        # Real D04 inference, then force one box to exercise the real RGB classifier.
+        result = det.detect(image)
+        assert isinstance(result, list)
+        outputs = [np.full((1,6,s,s), -30, np.float32) for s in (80,40,20)]
+        outputs[0][0,:,40,40] = [0,0,np.log(40),np.log(40),20,20]
+        real_session = det._session
+        det._session = SimpleNamespace(run=lambda *_: outputs)
+        result = det.detect(image)
+        assert len(result) == 1 and result[0].class_prob is not None
+        assert (result[0].stage_name is not None) == (kind == 'c07')
+        det._session = real_session
+    finally:
+        det.close()
+
+
+def test_real_d04_matches_reference_resize_and_decode():
+    """Independent D04 reference math: stretch, logits, grid, exp, NMS, normalized boxes."""
+    model = ROOT/'models/d04_ms5_plant.onnx'
+    if not model.exists():
+        pytest.skip('Local D04 weights are not installed')
+    det = OnnxDetector(DetectorSettings(model_path=model, classes_path=model.with_suffix('.json')),
+                       providers=['CPUExecutionProvider'])
+    try:
+        det.load()
+        image = np.random.default_rng(42).random((120,200,5), dtype=np.float32)
+        reference_input = np.stack([cv2.resize(image[...,c], (640,640)) for c in range(5)])[None]
+        raw = det._session.run(None, {det._input_name: reference_input})
+        boxes, scores = [], []
+        for output, stride in zip(raw, (8,16,32)):
+            reg = output[0,:4]
+            score = (1/(1+np.exp(-np.clip(output[0,4],-80,80)))) * (1/(1+np.exp(-np.clip(output[0,5],-80,80))))
+            y, x = np.indices(score.shape)
+            cx, cy = (reg[0]+x)*stride, (reg[1]+y)*stride
+            w, h = np.exp(reg[2])*stride, np.exp(reg[3])*stride
+            boxes.append(np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1).reshape(-1,4))
+            scores.append(score.ravel())
+        boxes = np.concatenate(boxes).astype(np.float32)
+        scores = np.concatenate(scores)
+        # Always exercise actual boxes even if the synthetic image has no confident plants.
+        det.confidence_threshold = float(np.max(scores)) * .90
+        selected = scores >= det.confidence_threshold
+        boxes, scores = boxes[selected], scores[selected]
+        xywh = boxes.copy(); xywh[:,2:] -= xywh[:,:2]
+        indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), det.confidence_threshold,
+                                  det.nms_iou_threshold)
+        keep = np.asarray(indices).reshape(-1)
+        expected = np.clip(boxes[keep]/640.0, 0, 1)
+        expected[:,2:] -= expected[:,:2]
+        actual = det.detect(image)
+        assert len(actual) == len(expected) > 0
+        np.testing.assert_allclose([[d.bbox.x,d.bbox.y,d.bbox.w,d.bbox.h] for d in actual], expected, atol=2e-6)
+        np.testing.assert_allclose([d.confidence for d in actual], scores[keep], atol=6e-5)
+    finally:
+        det.close()
+
+
+def test_wrong_onnx_input_is_rejected(tmp_path):
+    from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx
+    model = create_dummy_yolo_onnx(tmp_path/'rgb.onnx')
+    det = OnnxDetector(DetectorSettings(model_path=model, input_profile='ms5'), providers=['CPUExecutionProvider'])
+    with pytest.raises(DetectionError, match='MS5 ожидает'):
+        det.load()
+    assert not det.is_loaded
+
+
+def test_classifier_failure_is_not_silently_reported_as_plant(tmp_path):
+    det, _ = synthetic_detector(tmp_path)
+    def fail(*args):
+        raise RuntimeError('classifier failure')
+    det._classifier_session = SimpleNamespace(run=fail)
+    with pytest.raises(DetectionError, match='классификации'):
+        det.detect(np.full((100,200,5), .5, np.float32))
+
+
+def test_standalone_main_window_import_has_no_cycle():
+    import os
+    import subprocess
+    import sys
+    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
+    subprocess.run([sys.executable, '-c', 'from geo_weed_station.ui.main_window import MainWindow'],
+                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=20)
```

### `tests/controllers/test_offline_controller.py`

```diff
--- a/tests/controllers/test_offline_controller.py
+++ b/tests/controllers/test_offline_controller.py
@@ -424,3 +424,71 @@
 
 
 
+
+
+def test_ms5_selection_batch_preview_and_persistence(offline_setup, qtbot):
+    from pathlib import Path
+    from tests.test_ms5_integration import write_ms5
+    from geo_weed_station.schemas.session import FileEntry
+    from geo_weed_station.config.settings import DetectorSettings
+
+    ctrl, map_ctrl, window, tmp_path = offline_setup
+    qtbot.addWidget(window)
+    model = Path(__file__).resolve().parents[2] / 'models/d04_ms5_plant.onnx'
+    ctrl._on_model_changed(model)
+    assert ctrl._settings.detector.input_profile == 'ms5'
+    assert ctrl.filter_panel.selected_classifier_type() == 'none'
+    saved = DetectorSettings.model_validate_json(window._qsettings.value('offline/detector'))
+    assert saved.model_path == model
+    path = write_ms5(tmp_path/'frame.tif', np.full((40,80,5), .5, np.float32))
+    ctrl.file_list_panel.set_entries([FileEntry(path=str(path))])
+    ctrl._load_entry_to_viewer(ctrl.file_list_panel.model.entries[0])
+    qtbot.waitUntil(lambda: ctrl.image_viewer._pixmap_item is not None, timeout=5000)
+    ctrl.start_batch()
+    assert not ctrl.filter_panel._combo_model.isEnabled()
+    assert not ctrl.filter_panel._combo_classifier.isEnabled()
+    # A direct signal must not switch the active configuration mid-batch either.
+    ctrl._on_classifier_changed('c06')
+    assert ctrl._settings.detector.classifier_type == 'none'
+    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=15000)
+    assert ctrl.filter_panel._combo_model.isEnabled()
+    assert str(path) in ctrl._detections_by_path
+    assert not map_ctrl.engine.weed_map.points  # No XMP: preview/detection only.
+    ctrl._on_classifier_changed('c07')
+    assert not ctrl._detections_by_path
+    assert ctrl._settings.detector.classifier_type == 'c07'
+    window.shutdown_threads()
+
+
+def test_ms5_rejects_rgb_in_batch_with_visible_error(offline_setup, qtbot):
+    from pathlib import Path
+    from tests.helpers.make_media import make_real_jpeg
+    from geo_weed_station.schemas.session import FileEntry, ProcessingStatus
+    ctrl, _, window, tmp_path = offline_setup
+    qtbot.addWidget(window)
+    model = Path(__file__).resolve().parents[2]/'models/d04_ms5_plant.onnx'
+    ctrl._on_model_changed(model)
+    path = make_real_jpeg(tmp_path/'rgb.jpg')
+    ctrl.file_list_panel.set_entries([FileEntry(path=str(path))])
+    ctrl.start_batch()
+    qtbot.waitUntil(lambda: ctrl._batch_worker is None, timeout=10000)
+    assert ctrl.file_list_panel.model.entries[0].status == ProcessingStatus.ERROR
+    assert 'ошибок 1' in window.status_bar_panel.message_text
+    assert ctrl.filter_panel._combo_model.isEnabled()
+    window.shutdown_threads()
+
+
+def test_ms5_settings_restored(offline_setup, qtbot):
+    from pathlib import Path
+    ctrl, map_ctrl, window, _ = offline_setup
+    qtbot.addWidget(window)
+    model = Path(__file__).resolve().parents[2]/'models/d04_ms5_plant.onnx'
+    ctrl._on_model_changed(model)
+    ctrl._on_classifier_changed('c06')
+    restored_window = MainWindow(qsettings=window._qsettings)
+    qtbot.addWidget(restored_window)
+    restored = OfflineController(window=restored_window, map_ctrl=map_ctrl, settings=AppSettings())
+    assert restored._settings.detector.model_path == model
+    assert restored._settings.detector.input_profile == 'ms5'
+    assert restored.filter_panel.selected_classifier_type() == 'c06'
+    assert 'D04 MS5' in restored.filter_panel._combo_model.currentText()
```

### `tests/test_onnx_classification.py`

```diff
--- a/tests/test_onnx_classification.py
+++ b/tests/test_onnx_classification.py
@@ -129,7 +129,7 @@
 
 def test_real_resnet50_dualhead_c07_onnx_inference():
     """Verify inference on real resnet50_dualhead_c07.onnx if file is present."""
-    model_path = Path("/home/user/maximus/Ekat2026/flying_robotics/repositories/models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx")
+    model_path = Path(__file__).resolve().parents[1] / "models/c07_rgb_geosaero/c07_rgb_geosaero_best.onnx"
     if not model_path.is_file():
         pytest.skip("resnet50_dualhead_c07.onnx not found")
 
@@ -163,7 +163,7 @@
         Path("GeoWeedStation/models/resnet50_dualhead_c07.onnx"),
         Path("models/resnet50_dualhead_c07.onnx"),
         Path("models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx"),
-        Path("/home/user/maximus/Ekat2026/flying_robotics/repositories/models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx"),
+        Path(__file__).resolve().parents[1] / "models/c07_rgb_geosaero/c07_rgb_geosaero_best.onnx",
     ]
     cls_model = next((c for c in cands if c.is_file()), None)
     if cls_model is None:
```

### `tests/test_map_canvas_visual_fixes.py`

```diff
--- a/tests/test_map_canvas_visual_fixes.py
+++ b/tests/test_map_canvas_visual_fixes.py
@@ -55,7 +55,7 @@
     assert len(station_items) == 2
 
 
-def test_mosaic_footprint_on_demand_thumbnail_and_no_center_dot(tmp_path):
+def test_mosaic_footprint_on_demand_thumbnail_and_no_center_dot(tmp_path, qtbot):
     sample_img = make_geotagged_jpeg(tmp_path / "valid_000.jpg")
     fp = FrameFootprint(
         frame_id="valid_000",
@@ -69,6 +69,7 @@
         polygon=[(-10.0, 7.5), (10.0, 7.5), (10.0, -7.5), (-10.0, -7.5)],
     )
     item = MosaicFootprintItem(fp, pixmap=None)
+    qtbot.waitUntil(lambda: any(isinstance(it, QGraphicsPixmapItem) for it in item.childItems()), timeout=5000)
     pix_items = [it for it in item.childItems() if isinstance(it, QGraphicsPixmapItem)]
     assert len(pix_items) == 1
     assert not pix_items[0].pixmap().isNull()
@@ -246,3 +247,17 @@
 
 
 
+
+
+def test_ms5_mosaic_thumbnail_uses_rgb_in_background(tmp_path, qtbot):
+    import numpy as np
+    from tests.test_ms5_integration import write_ms5
+    image = np.broadcast_to(np.array([1., .5, 0., .9, .8], np.float32), (20,30,5)).copy()
+    path = write_ms5(tmp_path/'ms5.tif', image)
+    fp = FrameFootprint(frame_id='ms5', source_path=str(path), center_x=0, center_y=0,
+                        width_m=20, height_m=15, altitude_m=20, yaw_deg=0,
+                        polygon=[(-10,7.5),(10,7.5),(10,-7.5),(-10,-7.5)])
+    item = MosaicFootprintItem(fp)
+    qtbot.waitUntil(lambda: any(isinstance(it, QGraphicsPixmapItem) for it in item.childItems()), timeout=5000)
+    pix = next(it for it in item.childItems() if isinstance(it, QGraphicsPixmapItem))
+    assert pix.pixmap().toImage().pixelColor(0,0).getRgb() == (255,127,0,255)
```

В `docs/tasks/README.md` добавлена задача 21 с ссылкой на этот отчёт и статусом `[Completed]`.
