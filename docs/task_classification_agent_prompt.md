# Промт для агента: Интеграция этапа нейросетевой классификации видов и стадий растений (GeoWeedStation)

## 📌 Миссия агента
Твоя задача — внедрить этап нейросетевой классификации сорной растительности (видов и фенологических фаз развития) в настольную наземную станцию управления `GeoWeedStation`.
Сейчас система выполняет только одностадийную детекцию объектов через ONNX Runtime (`OnnxDetector`). Необходимо реализовать полноценный двухстадийный пайплайн инференса:
1. **Первая стадия (Детекция):** Нахождение очагов сорняков детектором (`d01_rgb_plant.onnx`, `weed_detection_yolo.onnx` или `yolov8m.onnx`) с формированием рамок и постобработкой NMS.
2. **Вторая стадия (Классификация кропов):** Вырезание патчей растений с 10% запасом по полям, приведение к размеру $224 \times 224$ (NCHW), нормализация ImageNet и запуск двухголовой классификационной модели ResNet-50 (`resnet50_dualhead_c07.onnx`) для определения категории сорняка (11 классов) и фазы его развития (5 стадий).

Реализация должна быть полностью совместима с эталонной логикой из `repositories/skyris/example/pipeline_orange_pi.py` и спецификацией из `repositories/models/MODELS.md`.

---

## 🧭 Контекст и окружение вокруг агента

### 1. Рабочая директория и виртуальное окружение
- **Корень проекта:** `/home/user/maximus/Ekat2026/flying_robotics/repositories/GeoWeedStation`
- **Виртуальное окружение (ОБЯЗАТЕЛЬНО):**
  Согласно `Agents.md`, для GeoWeedStation используется виртуальное окружение `.venv_gws`:
  ```bash
  source /home/user/maximus/Ekat2026/flying_robotics/repositories/.venv_gws/bin/activate
  # или прямой вызов:
  /home/user/maximus/Ekat2026/flying_robotics/repositories/.venv_gws/bin/python3
  /home/user/maximus/Ekat2026/flying_robotics/repositories/.venv_gws/bin/pytest
  ```

### 2. Расположение моделей
- Локальная папка моделей проекта: `GeoWeedStation/models/`
- Центральный репозиторий моделей: `/home/user/maximus/Ekat2026/flying_robotics/repositories/models`
  - Классификатор C07:
    `/home/user/maximus/Ekat2026/flying_robotics/repositories/models/c07_rgb_geosaero/resnet50_dualhead_c07.onnx`
    *(Размер ~89.8 МБ, ONNX Opset 12)*
  - Детектор D01:
    `/home/user/maximus/Ekat2026/flying_robotics/repositories/models/d01-rgb-plant/yolox_m.onnx`
    (в локальной папке также имеется `models/d01_rgb_plant.onnx`)

### 3. Архитектура проекта GeoWeedStation
- `geo_weed_station/schemas/entities.py`: Базовые сущности предметной области (`DetectedWeed`, `BBox`, `CoordinateSystem`).
- `geo_weed_station/core/detector_base.py`: Интерфейс детектора `BaseDetector`.
- `geo_weed_station/core/onnx_detector.py`: Реализация детектора на ONNX Runtime.
- `geo_weed_station/config/settings.py` и `default_config.json`: Настройки приложения (`DetectorSettings`).
- `geo_weed_station/controllers/offline_controller.py`: Контроллер офлайн-обработки фото и видео.
- `geo_weed_station/controllers/runs_controller.py`: Контроллер пакетных сессий.
- `geo_weed_station/core/annotation_io.py`: Импорт/экспорт аннотаций (JSON, GeoJSON, CSV).
- `geo_weed_station/ui/widgets/image_viewer.py` и `video_player_widget.py`: Визуализация рамок и меток.

---

## 🧠 Спецификация классификатора ONNX и математика предобработки

### 1. Архитектура модели `resnet50_dualhead_c07.onnx`
- **Входной тензор:**
  - Имя: `images`
  - Форма: `[1, 3, 224, 224]` (формат **NCHW**, `float32`).
- **Выходные тензоры:**
  - `logits_category`: `[1, 11]` (`float32`) — логиты 11 категорий сорняков GeosAero.
  - `logits_stage`: `[1, 5]` (`float32`) — логиты 5 фенологических стадий развития.

### 2. Предобработка входного кропа (Строго по pipeline_orange_pi.py и MODELS.md)
Для каждого обнаруженного детектором бокса в пикселях исходного кадра $[x_1, y_1, x_2, y_2]$:
1. Вычисление $10\%$ запаса (контекст вокруг растения):
   $$pad_w = 0.10 \cdot (x_2 - x_1), \quad pad_h = 0.10 \cdot (y_2 - y_1)$$
   $$cx_1 = \max(0, \operatorname{round}(x_1 - pad_w)), \quad cy_1 = \max(0, \operatorname{round}(y_1 - pad_h))$$
   $$cx_2 = \min(W_{orig}, \operatorname{round}(x_2 + pad_w)), \quad cy_2 = \min(H_{orig}, \operatorname{round}(y_2 + pad_h))$$
2. Валидация размера кропа: если $cx_2 - cx_1 < 4$ или $cy_2 - cy_1 < 4$ — пропуск или присвоение дефолтного класса.
3. Вырезка кропа: `crop_bgr = image_bgr[cy1:cy2, cx1:cx2]`.
4. Приведение цветового пространства: BGR $\to$ RGB (`cv2.cvtColor`).
5. Масштабирование: `cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)`.
6. Нормализация ImageNet:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   $$x_{norm} = \frac{crop\_rgb / 255.0 - \mu}{\sigma}$$
7. Транспонирование HWC $\to$ CHW и добавление batch:
   `tensor = np.transpose(x_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)`.

### 3. Декодирование логитов (Softmax)
$$p_{cat} = \operatorname{softmax}(logits\_category), \quad cat\_id = \operatorname{argmax}(p_{cat})$$
$$p_{stage} = \operatorname{softmax}(logits\_stage), \quad stage\_id = \operatorname{argmax}(p_{stage})$$

### 4. Таксономия C07
```python
C07_CATEGORIES = [
    "background",
    "Осот, Бодяк, Латук",
    "Вьюнок",
    "Щирица (виды)",
    "Маревые (Марь, Лебеда, Кохия)",
    "Канатник, просвирник, гибискус",
    "Крестоцветные (Капустные)",
    "Горец (виды)",
    "Злаковые",
    "Полынь, Ромашка",
    "Гвоздичные"
]

C07_STAGES = [
    "всходы",
    "2-4 листа",
    "взрослое растение",
    "переросшие",
    "не определена"
]
```

---

## 📋 Пошаговый план реализации для агента

### Шаг 1. Расширение доменной модели (`geo_weed_station/schemas/entities.py`)
1. В класс `DetectedWeed` добавить опциональные поля:
   ```python
   stage_name: str | None = None
   class_prob: float | None = None
   ```
2. Поскольку у `DetectedWeed` задано `model_config = ConfigDict(frozen=True, extra="forbid")`, добавление этих полей позволит безопасно хранить результат классификации без нарушения существующих контрактов.

### Шаг 2. Конфигурация детектора (`geo_weed_station/config/settings.py`, `default_config.json`)
1. В `DetectorSettings` добавить параметры:
   - `classifier_type: str = "none"` (допустимые значения: `"none"`, `"c07"`, `"c06"`).
   - `classifier_path: Path | None = None` (по умолчанию `None` или путь к `resnet50_dualhead_c07.onnx`).
   - `classifier_input_size: tuple[int, int] = (224, 224)`.
2. В функции резолвинга путей к моделям предусмотреть автоматический поиск `resnet50_dualhead_c07.onnx` в `models/` или центральном репозитории `../models/c07_rgb_geosaero/`.

### Шаг 3. Интеграция классификатора в `geo_weed_station/core/onnx_detector.py`
1. **Класс `OnnxClassifier` (или встроенный модуль в `OnnxDetector`):**
   - Ленивая загрузка ONNX-сессии классификатора при `classifier_type != "none"`.
   - Метод `classify_crops(image: np.ndarray, boxes_xyxy: np.ndarray) -> list[tuple[str, str, float]]`.
2. **Обновление метода `OnnxDetector.detect(image: np.ndarray) -> list[DetectedWeed]`:**
   - После этапа детекции и NMS, если классификатор активен:
     - Для каждого сохраненного бокса извлечь кроп растения.
     - Выполнить классификацию через сессию `resnet50_dualhead_c07.onnx`.
     - Заменить класс по умолчанию (`weed` / `weed_dicot`) на предсказанную таксономическую группу (например, `"Осот, Бодяк, Латук"`).
     - Заполнить `stage_name` (например, `"всходы"` или `"2-4 листа"`) и `class_prob`.
   - Если классификатор выключен (`classifier_type == "none"`), поведение должно строго соответствовать текущему (детекция без кроп-классификации).
3. **Метод `close()`:**
   - Освобождать сессии детектора и классификатора.

### Шаг 4. Поддержка сохранения и сериализации (`geo_weed_station/core/annotation_io.py`)
1. При экспорте в JSON / GeoJSON / CSV сохранять поля `stage_name` и `class_prob`.
2. При парсинге существующих файлов аннотаций сохранять обратную совместимость (если полей нет в старых файлах — выставлять `None`).

### Шаг 5. Отображение в пользовательском интерфейсе (`ui/widgets/`)
1. **`image_viewer.py` и `video_player_widget.py`:**
   - При наведении курсора или отображении бейджей рамок показывать не только имя класса и точность, но и стадию:
     `"{class_name} {conf:.2f} [{stage_name}]"`
   - Текст на русском языке в Qt рисуется через `QPainter` нативно, поэтому кодировка UTF-8 поддерживается из коробки.
2. **`detection_filter_panel.py`:**
   - Убедиться, что новые классы сорняков корректно появляются в дереве фильтрации или выпадающем списке категорий.

### Шаг 6. Написание тестов
1. Создать `tests/test_onnx_classification.py`:
   - Тест функции предобработки кропа (размерность $1 \times 3 \times 224 \times 224$, нормализация ImageNet).
   - Тест декодирования логитов C07 (softmax категорий и стадий).
   - Тест работы `OnnxDetector` с подключенным классификатором (с использованием реальной модели или синтетической dummy ONNX сессии).
2. Запуск тестов:
   ```bash
   /home/user/maximus/Ekat2026/flying_robotics/repositories/.venv_gws/bin/pytest tests/test_onnx_classification.py tests/test_onnx_detector.py -v
   ```

---

## 🎯 Критерии приемки (Definition of Done)
- [ ] В виртуальном окружении `.venv_gws` все тесты выполняются без ошибок.
- [ ] Модель `resnet50_dualhead_c07.onnx` успешно загружается в `OnnxDetector`.
- [ ] Кропы вырезаются с $10\%$ запасом, приводятся к $224 \times 224$ (NCHW) и нормализуются по стандарту ImageNet.
- [ ] Каждому детектированному сорняку присваивается русскоязычное название таксономии C07 и стадия развития.
- [ ] Аннотации с классификацией сохраняются и читаются через `annotation_io.py`.
- [ ] UI корректно визуализирует виды и стадии на снимках и видео.
