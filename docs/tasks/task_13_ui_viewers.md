# TASK 13 — UI: просмотр изображений и видеоплеер

## ID и название
`T-13` · Виджеты `image_viewer` и `video_player_widget` + список файлов пакета.

## Зависимости (Prerequisites)
T-02, T-11, T-12.

## Целевые файлы
- `src/geo_weed_station/ui/widgets/image_viewer.py`
- `src/geo_weed_station/ui/widgets/video_player_widget.py`
- `src/geo_weed_station/ui/widgets/file_list_panel.py` (таблица файлов: имя, размер, статус, XMP)
- `src/geo_weed_station/ui/widgets/detection_filter_panel.py`
- `src/geo_weed_station/ui/models/file_table_model.py` (`QAbstractTableModel`)
- `tests/ui/test_image_viewer.py`, `tests/ui/test_video_player_widget.py`, `tests/ui/test_file_list_panel.py`

## Интерфейсные контракты
```python
class DetectionFilterState(BaseModel):
    show_boxes: bool; show_labels: bool; show_confidence: bool
    min_confidence: float; hidden_classes: frozenset[str]

class ImageViewer(QGraphicsView):
    zoom_changed = Signal(float)
    weed_clicked = Signal(str)                         # weed_id
    def set_image(self, image: np.ndarray) -> None: ...
    def set_detections(self, detections: Sequence[DetectedWeed]) -> None: ...
    @Slot(DetectionFilterState)
    def apply_filter(self, state: DetectionFilterState) -> None: ...
    def fit_to_window(self) -> None: ...
    def clear(self) -> None: ...

class DetectionFilterPanel(QWidget):
    filter_changed = Signal(DetectionFilterState)
    def set_classes(self, names: Sequence[str]) -> None: ...

class VideoPlayerWidget(QWidget):
    play_requested = Signal(); pause_requested = Signal()
    seek_requested = Signal(int)
    frame_skip_changed = Signal(int)
    @Slot(object, list)
    def show_frame(self, frame: np.ndarray, detections: list[DetectedWeed]) -> None: ...
    @Slot(int, int)
    def set_position(self, current: int, total: int) -> None: ...

class FileTableModel(QAbstractTableModel):
    def set_entries(self, entries: list[FileEntry]) -> None: ...
    def update_status(self, row: int, status: ProcessingStatus) -> None: ...
class FileListPanel(QWidget):
    file_selected = Signal(int); process_requested = Signal()
```
Поведение: pan — drag, zoom — колесо с центрированием на курсоре, масштаб с сохранением aspect ratio; рамки рисуются как `QGraphicsItem` (или в буфере кадра для видео) — без блокирующих операций; изменения фильтра применяются мгновенно без повторного инференса.

## Критерии приемки (Definition of Done)
- [x] Zoom/pan плавные, масштаб центрируется относительно курсора; fit-to-window сохраняет пропорции.
- [x] Переключатели рамок/подписей/confidence и порог/классы корректно скрывают элементы.
- [x] Плеер: play/pause/seek/таймлайн и выбор пропуска кадров; UI ≥ 30 FPS на тестовом 720p-ролике (замер в тесте с допуском).
- [x] Таблица файлов отображает имя/размер/статус/признак XMP, обновляется по сигналам воркера.
- [x] Виджеты не загружают файлы и не вызывают инференс — только отображают данные.
- [x] Изображение 8000×6000 отображается без переполнения памяти (даунскейл для сцены).

## План верификации / Тесты
`pytest-qt` offscreen: масштаб колесом (проверка transform/anchor), фильтры → количество видимых `QGraphicsItem`, эмиссия сигналов плеера, обновление модели таблицы, граничные значения таймлайна. Скриншотные снимки (`widget.grab()`) для sanity-проверки.
