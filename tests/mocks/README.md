# Agrovision Onboard Mock Server

Эмулятор бортового комплекса Agrovision Onboard для локальной отладки и тестирования GeoWeedStation без реального БПЛА.

## Запуск из командной строки

```bash
# Стандартный запуск на порту 8000 с шагом автомата 0.2 с:
python tests/mocks/mock_onboard_server.py --port 8000 --tick 0.2

# Запуск в режиме имитации сбоев (timeout, drop, http500):
python tests/mocks/mock_onboard_server.py --port 8000 --fail-mode timeout
python tests/mocks/mock_onboard_server.py --port 8000 --fail-mode drop
python tests/mocks/mock_onboard_server.py --port 8000 --fail-mode http500
```

## Использование в тестах (pytest)

Сервер предоставляет фабричную функцию `run_in_thread(config: MockConfig)` и фикстуру `mock_server` в `tests/conftest.py`:

```python
def test_something(mock_server):
    # mock_server.url содержит адрес, например: http://127.0.0.1:41235
    client = DroneClient(base_url=mock_server.url)
    status = client.get_status()
    assert status.flight_state == FlightState.IDLE
```

## Эндпоинты
- `GET /status` — телеметрия дрона, текущее состояние (`idle`, `takeoff`, `flying_to_wp`, `processing`, `landing`, `completed`, `aborted`).
- `POST /start` — запуск миссии (`StartRequest` -> `StartResponse`, статус 202). Код 409 при активном полёте.
- `POST /stop` — прерывание миссии (`land` / `rth`).
- `GET /runs` — список сохранённых проходов (`DroneRun`).
- `GET /runs/{id}/data?type=summary|archive|image&wp_index=` — скачивание данных миссии (JSON-сводка, ZIP со снимками и XMP, единичный кадр).
- `DELETE /runs/{id}` — удаление данных прохода. Код 409 при удалении активной миссии.
