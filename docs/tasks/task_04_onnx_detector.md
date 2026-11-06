# TASK 04 — Детектор на ONNX Runtime

## ID и название
`T-04` · Абстрактный детектор, ONNX-реализация (letterbox, NMS, ленивая загрузка).

## Зависимости (Prerequisites)
T-01, T-02.

## Целевые файлы
- `src/geo_weed_station/core/detector_base.py` (ABC `BaseDetector`)
- `src/geo_weed_station/core/onnx_detector.py`
- `src/geo_weed_station/core/image_io.py` (безопасное чтение изображений через OpenCV, ограничение размера до 8000×6000)
- `tests/test_onnx_detector.py`
- `tests/helpers/make_dummy_onnx.py` (генерация крошечной ONNX-модели с фиксированным выходом для тестов)

## Интерфейсные контракты
```python
class BaseDetector(ABC):
    @abstractmethod
    def load(self) -> None: ...
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedWeed]: ...
    @abstractmethod
    def close(self) -> None: ...
    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

@dataclass(frozen=True)
class LetterboxResult:
    tensor: np.ndarray; scale: float; pad: tuple[float, float]

def letterbox(image: np.ndarray, size: tuple[int, int]) -> LetterboxResult: ...
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]: ...
def load_class_names(path: Path) -> dict[int, str]: ...

class OnnxDetector(BaseDetector):
    def __init__(self, settings: DetectorSettings, providers: Sequence[str] | None = None) -> None: ...
    def detect(self, image: np.ndarray) -> list[DetectedWeed]: ...   # нормализованные BBox 0..1 в системе исходного кадра
    def set_thresholds(self, confidence: float, iou: float) -> None: ...
    @staticmethod
    def available_providers() -> list[str]: ...

def read_image(path: Path) -> np.ndarray: ...   # ImageReadError при повреждении
```
Ленивая инициализация `InferenceSession`: только при первом `detect()`/`load()`. Автовыбор провайдера: аппаратный при наличии, иначе `CPUExecutionProvider`.

## Критерии приемки (Definition of Done)
- [ ] Модуль зависит только от numpy/cv2/onnxruntime и `schemas`; нет PySide6.
- [ ] Letterbox сохраняет aspect ratio; обратное преобразование координат корректно (ошибка ≤ 1 px).
- [ ] Фильтр по confidence и NMS работают по порогам из настроек.
- [ ] `close()` освобождает сессию; повторный `load()` допустим.
- [ ] Повреждённый файл → `ImageReadError`, без падения процесса.
- [ ] Изображение 8000×6000 обрабатывается без превышения лимита памяти (контролируемый downscale перед тензором).
- [ ] Лицензии: используется только ONNX Runtime (MIT); нет AGPL (ultralytics запрещён).

## План верификации / Тесты
`tests/test_onnx_detector.py`: letterbox туда-обратно; NMS на синтетических боксах; dummy-ONNX возвращает известные боксы → проверка классов/координат; пороги; ленивая загрузка (сессия не создана до `detect`); `close()`; битый JPEG; большой кадр (маркер `slow`).
