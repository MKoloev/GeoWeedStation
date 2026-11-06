# TASK 11 — Фоновые воркеры офлайн-режима (пакетная обработка и видео)

## ID и название
`T-11` · `batch_image_worker`, `video_worker`, пул `QRunnable` для XMP-индексации, фабрика источников данных.

## Зависимости (Prerequisites)
T-02, T-03, T-04, T-05, T-06, T-10 (базовый `ManagedWorker`).

## Целевые файлы
- `src/geo_weed_station/workers/batch_image_worker.py`
- `src/geo_weed_station/workers/video_worker.py`
- `src/geo_weed_station/workers/xmp_index_task.py` (`QRunnable` для пула)
- `src/geo_weed_station/workers/frame_queue.py` (потокобезопасная очередь кадров с ограничением длины)
- `src/geo_weed_station/core/source_factory.py` (Factory: single / folder / video / drone_run)
- `src/geo_weed_station/core/session.py` (`ProcessingSession`, автосохранение)
- `src/geo_weed_station/schemas/session.py` (`FileEntry`, `ProcessingStatus`, `SessionState`)
- `tests/test_batch_image_worker.py`, `tests/test_video_worker.py`, `tests/test_source_factory.py`, `tests/test_session.py`

## Интерфейсные контракты
```python
class ProcessingStatus(StrEnum): PENDING; PROCESSING; DONE; ERROR; NO_METADATA
class FileEntry(BaseModel): path; size_bytes; status; has_xmp: bool; detections_count: int | None

class SourceFactory:
    @staticmethod
    def create(kind: Literal["image","folder","video","drone_run"], path: Path) -> "FrameSource": ...
class FrameSource(Protocol):
    def entries(self) -> list[FileEntry]: ...
    def load(self, entry: FileEntry) -> FramePayload: ...

class ProcessingSession:
    entries: list[FileEntry]
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "ProcessingSession": ...

class XmpIndexTask(QRunnable):                     # результаты через QObject-сигнал-обёртку
    ...

class BatchImageWorker(ManagedWorker):
    progress = Signal(int, int, str)               # i, N, текст статуса
    frame_processed = Signal(FramePayload)
    entry_status_changed = Signal(int, ProcessingStatus)
    batch_finished = Signal(int, int)              # успешно, ошибок
    @Slot(list)
    def process(self, entries: list[FileEntry], run_inference: bool) -> None: ...
    @Slot()
    def cancel(self) -> None: ...

class VideoWorker(ManagedWorker):
    frame_ready = Signal(object, list)             # ndarray (копия) + list[DetectedWeed]
    position_changed = Signal(int, int)            # кадр, всего
    @Slot(str) def open(self, path: str) -> None: ...
    @Slot() def play(self) -> None: ...
    @Slot() def pause(self) -> None: ...
    @Slot(int) def seek(self, frame_index: int) -> None: ...
    @Slot(int) def set_frame_skip(self, n: int) -> None: ...
```
Правила: ONNX и `cv2.VideoCapture` живут только в потоках воркеров; модель инициализируется лениво; на промежуточных кадрах рамки удерживаются/интерполируются; ресурсы (VideoCapture, ONNX-сессия) освобождаются в `finally`; повреждённый файл → статус `ERROR`, пакет продолжается.

## Критерии приемки (Definition of Done)
- [ ] Рекурсивное сканирование поддерживает `.jpg/.jpeg/.png/.tif/.tiff`; индикатор XMP заполняется через пул потоков.
- [ ] Отмена пакета — безопасная, без утечек, промежуточные результаты сохраняются в сессию.
- [ ] Видео-воркер: `play/pause/seek`, пропуск кадров N, подавление дропа кадров очередью с ограничением.
- [ ] Повреждённые файлы не вызывают падения; `NO_METADATA` для файлов без XMP.
- [ ] Прогресс 0–100% и текст «инференс кадра N из M» эмитятся при каждом шаге.
- [ ] Нет разделяемой памяти между потоками: кадры передаются копиями.
- [ ] Ленивая загрузка ONNX проверена тестом (модель не загружена до первого инференса).

## План верификации / Тесты
`pytest-qt`: пакет из 10 синтетических JPEG (2 без XMP, 1 битый) → статусы и счётчики; отмена на середине; видео-тест на сгенерированном `.mp4` (скип кадров, seek, окончание); проверка закрытия ресурсов (mock `VideoCapture.release`). Тест производительности очереди кадров.
