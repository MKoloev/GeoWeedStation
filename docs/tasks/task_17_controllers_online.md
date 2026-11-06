# TASK 17 — Контроллеры: онлайн-режим и менеджер бортовых запусков

## ID и название
`T-17` · `OnlineController`, `RunsController` (подключение, миссия, телеметрия, live-карта, загрузка/вторичная обработка/удаление запусков).

## Зависимости (Prerequisites)
T-07, T-08, T-09, T-10, T-11, T-14, T-15, T-16.

## Целевые файлы
- `src/geo_weed_station/controllers/online_controller.py`
- `src/geo_weed_station/controllers/runs_controller.py`
- `src/geo_weed_station/controllers/app_controller.py` (подключение онлайн-контроллеров)
- `tests/controllers/test_online_controller.py`, `tests/controllers/test_runs_controller.py`

## Интерфейсные контракты
```python
class OnlineController(QObject):
    def __init__(self, window: MainWindow, map_ctrl: MapController, settings: AppSettings) -> None: ...
    @Slot(str, int) def connect_to(self, host: str, port: int) -> None: ...   # запуск TelemetryWorker
    @Slot() def disconnect(self) -> None: ...
    @Slot(PlannerParams) def plan(self, params: PlannerParams) -> None: ...  # через mission_planner → MapCanvas.set_route
    @Slot(MissionTask) def send_mission(self, task: MissionTask) -> None: ...  # требует flight_state == IDLE
    @Slot(StopAction) def stop(self, action: StopAction) -> None: ...
    @Slot(DroneStatus) def on_status(self, s: DroneStatus) -> None: ...      # обновление телеметрии и маркера
    # live: при INFERENCE_SEQUENTIAL периодически запрашивает summary → MapController.on_summary
    # при is_data_ready → итоговая выгрузка summary и разблокировка интерфейса

class RunsController(QObject):
    @Slot() def refresh(self) -> None: ...                  # GET /runs
    @Slot(str) def download(self, run_id: str) -> None: ...  # ZIP → распаковка → список кадров
    @Slot(str) def process_locally(self, run_id: str) -> None: ...   # BatchImageWorker + ONNX → MapController
    @Slot(str) def delete(self, run_id: str) -> None: ...    # после подтверждения; показ освобождённого объёма
```
Правила: отправка миссии — только после успешного `GET /status` с `idle` (клиентский сценарий из спецификации); во время полёта UI редактирования заблокирован, после `completed/aborted` — разблокируется; результат борта (`SummaryDetections`) попадает в `MapEngine` без повторной проекции; для миссий `COLLECTION_ONLY` предлагается вторичная локальная обработка скачанных снимков.

## Критерии приемки (Definition of Done)
- [x] Подключение: индикатор `CONNECTING → CONNECTED`, при обрыве — `RECONNECTING`, приложение остаётся отзывчивым.
- [x] Полный цикл с мок-сервером: connect → plan → start → мониторинг → live-карта → завершение → разблокировка.
- [x] `POST /stop` с `land` и `rth` корректно отрабатывает, интерфейс переходит к ожиданию выгрузки.
- [x] Менеджер запусков: список, скачивание с прогрессом/отменой, вторичная обработка, удаление с подтверждением и отображением `freed_bytes`.
- [x] Ошибки 409/400/404 показываются пользователю без падения.
- [x] Ограничение слота 10 минут: индикатор оставшегося времени/предупреждение при превышении.
- [x] Ни один сетевой вызов не выполняется в GUI-потоке (проверка тестом по `QThread.currentThread()` в мок-клиенте).

## План верификации / Тесты
`pytest-qt` + `mock_server` (T-09): полный онлайн-сценарий; `fail_mode=drop/timeout` → reconnect; конфликт 409; скачивание и локальная обработка архива (dummy-ONNX); удаление запуска; проверка запрета отправки при `flight_state != idle`.
