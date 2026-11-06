# TASK 08 — Сетевой слой: REST-клиент борта и менеджер загрузок

## ID и название
`T-08` · Автономная библиотека `drone_client`, исключения, `download_manager` (потоковая выгрузка ZIP).

## Зависимости (Prerequisites)
T-01, T-02.

## Целевые файлы
- `src/geo_weed_station/network/exceptions.py`
- `src/geo_weed_station/network/drone_client.py`
- `src/geo_weed_station/network/download_manager.py`
- `src/geo_weed_station/network/backoff.py` (экспоненциальная задержка)
- `tests/test_drone_client.py`, `tests/test_download_manager.py`, `tests/test_backoff.py`

## Интерфейсные контракты
```python
# exceptions.py
class DroneNetworkError(Exception): ...
class DroneTimeoutError(DroneNetworkError): ...
class DroneConnectionError(DroneNetworkError): ...
class DroneApiError(DroneNetworkError):   # status_code: int, detail: str   (400/404/409/500)
class DroneConflictError(DroneApiError): ...
class DroneNotFoundError(DroneApiError): ...

# drone_client.py  — синхронный, без Qt; вызывается ТОЛЬКО из воркеров
class DroneClient:
    def __init__(self, host: str, port: int, timeout_s: float = 2.0) -> None: ...
    def configure(self, host: str, port: int) -> None: ...
    def get_status(self) -> DroneStatus: ...                               # GET /status
    def start_mission(self, task: MissionTask) -> StartResponse: ...       # POST /start (202)
    def stop_mission(self, action: StopAction) -> StopResponse: ...        # POST /stop
    def list_runs(self) -> list[DroneRun]: ...                             # GET /runs
    def get_summary(self, run_id: str) -> SummaryDetections: ...           # GET /runs/{id}/data?type=summary
    def get_image(self, run_id: str, wp_index: int) -> bytes: ...          # ?type=image&wp_index=N
    def delete_run(self, run_id: str) -> DeleteRunResponse: ...            # DELETE /runs/{id}
    def stream_archive(self, run_id: str, chunk_bytes: int) -> Iterator[bytes]: ...  # ?type=archive
    def close(self) -> None: ...

# backoff.py
class ExponentialBackoff:
    def __init__(self, initial_s: float, max_s: float, factor: float) -> None: ...
    def next_delay(self) -> float: ...
    def reset(self) -> None: ...

# download_manager.py
class DownloadCancelled(Exception): ...
class DownloadManager:
    def __init__(self, client: DroneClient, target_dir: Path, chunk_bytes: int) -> None: ...
    def download_run(self, run_id: str, progress_cb: Callable[[int, int | None], None] | None = None,
                     cancel_flag: Callable[[], bool] | None = None) -> Path: ...  # скачать → .part → атомарный rename
    def extract_run(self, archive: Path) -> Path: ...                              # защита от zip-slip
    def list_images(self, run_dir: Path) -> list[Path]: ...
```
Ограничения: единственный транспорт — HTTP/REST; таймаут ≤ 2 с (ограничение в конструкторе); без sidecar-логики; ответы всегда проверяются Pydantic-моделями; ошибки HTTP маппятся в иерархию исключений.

## Критерии приемки (Definition of Done)
- [ ] Каждый метод соответствует эндпоинту спецификации `agrovision_onboard_rest_api_specification.md`.
- [ ] Таймаут и обрыв соединения → `DroneTimeoutError` / `DroneConnectionError`, не «сырые» исключения requests/httpx.
- [ ] Статусы 400/404/409/500 мапятся в типизированные исключения с телом ошибки.
- [ ] Архив качается чанками (память не растёт пропорционально размеру), прогресс-коллбэк и отмена работают, `.part` удаляется при отмене/ошибке.
- [ ] `extract_run` защищён от path traversal (zip-slip).
- [ ] Пакет `network` не импортирует PySide6 и `ui`/`workers`.
- [ ] `ExponentialBackoff` монотонно растёт до `max_s` и сбрасывается.

## План верификации / Тесты
Модульные тесты на `responses`/`respx` (или `pytest-httpserver`): все эндпоинты, коды ошибок, таймаут. Интеграционные тесты против `mock_onboard_server` добавляются после T-09 (маркер `integration`). Тест памяти потокового скачивания (синтетический ZIP ~50 МБ), тест zip-slip.
