# TASK 16 — Контроллеры: офлайн-режим и общий экспорт

## ID и название
`T-16` · `OfflineController` + общий `MapController` (связка View ↔ Workers ↔ MapEngine).

## Зависимости (Prerequisites)
T-03, T-04, T-06, T-11, T-12, T-13, T-14, T-15.

## Целевые файлы
- `src/geo_weed_station/controllers/__init__.py`
- `src/geo_weed_station/controllers/offline_controller.py`
- `src/geo_weed_station/controllers/map_controller.py`
- `src/geo_weed_station/controllers/export_controller.py`
- `src/geo_weed_station/controllers/app_controller.py` (создание контроллеров, жизненный цикл потоков)
- `main.py` (создание `AppController`)
- `tests/controllers/test_offline_controller.py`, `tests/controllers/test_map_controller.py`, `tests/controllers/test_export_controller.py`

## Интерфейсные контракты
```python
class MapController(QObject):
    map_updated = Signal(); statistics_changed = Signal(MapStatistics)
    map_locked_changed = Signal(bool, str)
    def __init__(self, engine: MapEngine) -> None: ...
    @Slot(FramePayload)
    def on_frame(self, payload: FramePayload) -> None: ...        # блокировка при has_georef()==False
    @Slot(SummaryDetections)
    def on_summary(self, summary: SummaryDetections) -> None: ...
    @Slot(float)
    def set_cell_size(self, cell_m: float) -> None: ...
    def reset(self) -> None: ...

class OfflineController(QObject):
    def __init__(self, window: MainWindow, map_ctrl: MapController, settings: AppSettings) -> None: ...
    @Slot() def open_image(self) -> None: ...
    @Slot() def open_folder(self) -> None: ...
    @Slot() def open_video(self) -> None: ...
    @Slot() def start_batch(self) -> None: ...
    @Slot() def cancel(self) -> None: ...
    @Slot(DetectionFilterState) def on_filter_changed(self, s: DetectionFilterState) -> None: ...

class ExportController(QObject):
    @Slot() def export(self) -> None: ...      # открывает ExportDialog → MapExporter; запрещён при блокировке карты

class AppController(QObject):
    def start(self) -> None: ...
    def shutdown(self) -> None: ...            # остановка всех потоков и ресурсов
```
Правила: контроллеры — единственное место, где соединяются сигналы виджетов и воркеров; бизнес-логики расчётов нет (она в `core/`); циклических импортов `ui ↔ workers` нет.

## Критерии приемки (Definition of Done)
- [x] Сценарий: выбор файла → отображение → детекция → рамки на `ImageViewer`.
- [x] Сценарий: папка → таблица → «Обработать» → прогресс/отмена → точки и heatmap на карте.
- [x] Видео: открытие → play → рамки в реальном времени → пауза/seek.
- [x] Отсутствие XMP/высоты блокирует вкладку карты с точным текстом предупреждения; экспорт и heatmap недоступны.
- [x] Экспорт CSV/GeoJSON/PNG выполняется через `ExportDialog` и `MapExporter`.
- [x] Офлайн-режим не создаёт сетевых соединений (проверка тестом с блокировкой сокетов).
- [x] Завершение приложения корректно останавливает воркеры.

## План верификации / Тесты
`pytest-qt` с реальными воркерами и dummy-ONNX: сквозной офлайн-сценарий на синтетическом наборе (с XMP и без); тест блокировки сокетов (`pytest-socket`); тест отмены; тест экспорта во временный каталог.
