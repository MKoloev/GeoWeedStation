# TASK 14 — UI: холст карты и панель управления миссией

## ID и название
`T-14` · Виджеты `map_canvas` и `mission_control` (планирование, режимы, управление полётом, телеметрия).

## Зависимости (Prerequisites)
T-02, T-05, T-06, T-07, T-12.

## Целевые файлы
- `src/geo_weed_station/ui/widgets/map_canvas.py`
- `src/geo_weed_station/ui/widgets/map_info_panel.py` (сводка метрик, попап точки с миниатюрой)
- `src/geo_weed_station/ui/widgets/mission_control.py`
- `src/geo_weed_station/ui/widgets/telemetry_panel.py`
- `src/geo_weed_station/ui/widgets/runs_table.py` (таблица бортовых запусков)
- `tests/ui/test_map_canvas.py`, `tests/ui/test_mission_control.py`, `tests/ui/test_runs_table.py`

## Интерфейсные контракты
```python
class MapCanvas(QGraphicsView):
    weed_selected = Signal(str)
    def set_coordinate_system(self, cs: CoordinateSystem) -> None: ...
    def set_points(self, points: Sequence[WeedPoint]) -> None: ...
    def append_points(self, points: Sequence[WeedPoint]) -> None: ...
    def set_heatmap(self, grid: HeatmapGrid | None) -> None: ...
    @Slot(int)
    def set_heatmap_opacity(self, percent: int) -> None: ...
    def set_route(self, waypoints: Sequence[Waypoint]) -> None: ...
    def set_drone_pose(self, pose: DronePose | None) -> None: ...   # маркер + вектор оптической оси
    def set_locked(self, locked: bool, message: str | None) -> None: ...
    def render_to_image(self, scale: float) -> np.ndarray: ...

class MapInfoPanel(QWidget):
    def set_statistics(self, stats: MapStatistics) -> None: ...
    def show_point(self, point: WeedPoint, thumbnail: np.ndarray | None) -> None: ...

class MissionControl(QWidget):
    plan_requested = Signal(PlannerParams)
    send_mission_requested = Signal(MissionTask)
    stop_requested = Signal(StopAction)
    def set_flight_locked(self, locked: bool) -> None: ...   # блок редактирования на время полёта
    def set_validation_errors(self, errors: list[str]) -> None: ...
class TelemetryPanel(QWidget):
    @Slot(DroneStatus)
    def update_status(self, status: DroneStatus) -> None: ...   # этап, точка N из M, X/Y/Z/Yaw, кадры

class RunsTable(QWidget):
    refresh_requested = Signal(); download_requested = Signal(str)
    delete_requested = Signal(str); process_requested = Signal(str)
    def set_runs(self, runs: list[DroneRun]) -> None: ...
```
`MissionControl`: ввод двух угловых точек, выбор СК, высоты, перекрытия; переключатель `COLLECTION_ONLY / INFERENCE_SEQUENTIAL`; флаги `save_images` / `save_metadata` (маппятся в `SavePolicy`); кнопки Старт, Land, Return (RTH); локальная валидация до отправки.

## Критерии приемки (Definition of Done)
- [x] Точечный слой окрашен по классам; клик по точке эмитит `weed_selected` и открывает панель с миниатюрой.
- [x] Heatmap — полупрозрачный градиент (тёмно-зелёный → красный), слайдер прозрачности работает.
- [x] Маршрут (змейка) отображается до отправки; маркер дрона обновляется ≤ 100 мс после `update_status`.
- [x] В заблокированном состоянии карта показывает предупреждение и не отрисовывает слои.
- [x] `MissionControl` блокирует отправку невалидных конфигураций (напр. `COLLECTION_ONLY` без сохранения изображений) и показывает ошибки.
- [x] Во время полёта поля редактирования заблокированы; доступны только Stop-действия.
- [x] Таблица запусков показывает run_id, дату, статус, размер (МБ), содержимое.
- [x] Виджеты не обращаются к сети/MapEngine напрямую (только сигналы/слоты и данные).

## План верификации / Тесты
`pytest-qt`: добавление 1000 точек (время отрисовки), клик/попап, opacity, locked-режим, генерация маршрута → количество точек на сцене, валидация конфликтов, блокировка при полёте, измерение задержки обновления маркера дрона.
