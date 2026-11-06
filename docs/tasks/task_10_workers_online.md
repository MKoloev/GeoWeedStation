# TASK 10 — Фоновые воркеры онлайн-режима (QThread)

## ID и название
`T-10` · `telemetry_worker`, `download_worker`, командный воркер, базовый класс воркеров.

## Зависимости (Prerequisites)
T-01, T-02, T-08, T-09 (для интеграционных тестов).

## Целевые файлы
- `src/geo_weed_station/workers/base_worker.py`
- `src/geo_weed_station/workers/telemetry_worker.py`
- `src/geo_weed_station/workers/command_worker.py` (start/stop/list_runs/get_summary/delete вне GUI-потока)
- `src/geo_weed_station/workers/download_worker.py`
- `tests/test_telemetry_worker.py`, `tests/test_download_worker.py`, `tests/test_command_worker.py`

## Интерфейсные контракты
```python
class ManagedWorker(QObject):            # общий жизненный цикл
    started_ok = Signal(); finished = Signal(); failed = Signal(str)
    def start_work(self) -> None: ...    # слот
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def request_stop(self) -> None: ...  # безопасное прерывание

class TelemetryWorker(ManagedWorker):
    status_received = Signal(DroneStatus)
    connection_changed = Signal(ConnectionState)
    latency_measured = Signal(float)
    def __init__(self, client: DroneClient, settings: NetworkSettings) -> None: ...
    @Slot(str, int)
    def set_endpoint(self, host: str, port: int) -> None: ...
    @Slot(float)
    def set_poll_rate(self, hz: float) -> None: ...

class CommandWorker(ManagedWorker):
    start_requested/stop_requested/runs_requested/summary_requested/delete_requested  # входные слоты
    mission_started = Signal(StartResponse); mission_stopped = Signal(StopResponse)
    runs_loaded = Signal(list); summary_loaded = Signal(SummaryDetections)
    run_deleted = Signal(DeleteRunResponse); command_failed = Signal(str, str)  # (command, message)

class DownloadWorker(ManagedWorker):
    progress = Signal(int, int)          # байт скачано / всего (−1 если неизвестно)
    download_finished = Signal(str)      # путь к распакованному каталогу
    @Slot(str)
    def download(self, run_id: str) -> None: ...
    @Slot()
    def cancel(self) -> None: ...

def start_in_thread(worker: ManagedWorker, name: str) -> QThread: ...
```
Правила: все сетевые вызовы — только внутри воркеров; обмен с GUI — исключительно сигналами/слотами с копируемыми/иммутабельными DTO; при обрыве — состояние `RECONNECTING` и ожидание `ExponentialBackoff`; опрос не накапливает запросы (если предыдущий не завершён, тик пропускается).

## Критерии приемки (Definition of Done)
- [ ] Опрос `/status` с частотой 1–2 Гц, таймаут запроса ≤ 2 с; GUI не блокируется.
- [ ] Последовательность состояний при обрыве: `CONNECTED → RECONNECTING → … → CONNECTED` с растущей задержкой.
- [ ] `request_stop()` завершает поток ≤ 3 с без утечек сокетов (проверяется `QThread.isFinished()`).
- [ ] Скачивание: прогресс и отмена; частичный файл удаляется при отмене.
- [ ] Воркеры не обращаются к виджетам; `workers/` не импортирует `ui/`.
- [ ] Один запрос в полёте на тик (нет гонки запросов).

## План верификации / Тесты
`pytest-qt` + `mock_server` (T-09): цикл подключения; режимы `drop/timeout` → reconnect; корректное завершение; скачивание архива с отменой; ошибки команд (409) приходят сигналом `command_failed`.
