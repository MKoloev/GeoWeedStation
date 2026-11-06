# TASK 09 — Мок-сервер бортового комплекса (эмулятор БПЛА)

## ID и название
`T-09` · `tests/mocks/mock_onboard_server.py`: FastAPI-эмулятор `agrovision_onboard` для автономной отладки онлайн-режима без физического дрона.

## Зависимости (Prerequisites)
T-02, T-08.

## Целевые файлы
- `tests/mocks/__init__.py`
- `tests/mocks/mock_onboard_server.py`
- `tests/mocks/fixtures_generator.py` (генерация тестовых JPEG с XMP, summary JSON и ZIP-архивов миссий)
- `tests/mocks/README.md` (запуск и сценарии)
- `tests/conftest.py` (фикстура `mock_server`, поднимающая uvicorn в фоновом потоке на свободном порту)
- `tests/test_mock_server_contract.py`, `tests/test_drone_client_integration.py`

## Интерфейсные контракты
```python
class MockConfig(BaseModel):
    tick_s: float = 0.2                  # длительность этапа конечного автомата
    fail_mode: Literal["none", "timeout", "drop", "http500"] = "none"
    latency_s: float = 0.0
    preset: str = "clever4_csi_onnx"
    initial_runs: int = 1

class MockDroneState:                    # потокобезопасное состояние
    def start(self, req: StartRequest) -> StartResponse: ...
    def stop(self, req: StopRequest) -> StopResponse: ...
    def snapshot(self) -> DroneStatus: ...
    def advance(self) -> None: ...       # IDLE→TAKEOFF→FLYING_TO_WP→PROCESSING→…→LANDING→COMPLETED/ABORTED

def create_app(config: MockConfig | None = None, state: MockDroneState | None = None) -> FastAPI: ...
def run_in_thread(config: MockConfig, port: int = 0) -> "MockServerHandle": ...   # .url, .stop()
if __name__ == "__main__": ...           # CLI: --port, --tick, --fail-mode
```
Эндпоинты (точное соответствие спецификации): `POST /start` (202, 409 при активном полёте, 400 при невалидном теле), `POST /stop` (`land`/`rth`), `GET /status`, `GET /runs`, `GET /runs/{id}/data?type=summary|archive|image&wp_index=`, `DELETE /runs/{id}` (409 для активной миссии, 404 для неизвестной). Ответы используют схемы из T-02. `summary` растёт по мере прохождения точек; `archive` — настоящий ZIP со снимками+XMP; поддержка `fail_mode` для проверки reconnect/backoff.

## Критерии приемки (Definition of Done)
- [ ] Сервер запускается командой `python tests/mocks/mock_onboard_server.py --port 8000` и в тестах как фикстура.
- [ ] Конечный автомат соответствует диаграмме состояний из спецификации; `flight_state` принимает только допустимые значения.
- [ ] Коды 200/202/400/404/409/500 возвращаются в описанных ситуациях.
- [ ] `is_data_ready` становится `true` только после `completed`/`aborted`.
- [ ] Скачанный ZIP содержит JPEG с XMP, разбираемые `xmp_parser` (T-03).
- [ ] Режимы `timeout/drop/http500` воспроизводимо вызывают соответствующие ошибки клиента.
- [ ] Весь контракт валидируется Pydantic-схемами из `schemas/onboard_api.py`.

## План верификации / Тесты
`test_mock_server_contract.py`: каждый эндпоинт × успех/ошибка. `test_drone_client_integration.py`: полный сценарий (status → start → polling до COMPLETED → summary → archive → delete), сценарий `/stop` (land и rth), конфликт 409, разрыв связи и восстановление.
