# TASK 01 — Базовая инфраструктура проекта и конфигурация

## ID и название
`T-01` · Скелет проекта, зависимости, Pydantic-настройки, логирование, точка входа-заглушка.

## Зависимости (Prerequisites)
Нет (стартовая задача).

## Целевые файлы
Корень: `/media/sf_repositories/GeoWeedStation/`
- `pyproject.toml` (метаданные пакета, `src`-layout, настройки pytest/ruff/mypy)
- `requirements.txt` (PySide6, onnxruntime, opencv-python-headless, numpy, pydantic>=2, pydantic-settings, requests (или httpx), pyproj, scipy, fastapi, uvicorn, pytest, pytest-qt, pytest-cov, pyinstaller, Pillow)
- `AGENTS.md` (правила для AI-ассистентов: слои, запрет кода в UI-потоке, лицензии)
- `main.py`
- `src/geo_weed_station/__init__.py` и `__init__.py` во всех подпакетах: `config/`, `schemas/`, `core/`, `core/map_engine/`, `network/`, `workers/`, `controllers/`, `ui/`, `ui/widgets/`, `ui/dialogs/`
- `src/geo_weed_station/config/settings.py`
- `src/geo_weed_station/config/default_config.json`
- `src/geo_weed_station/config/logging_setup.py`
- `models/classes.json` (ID класса → название; минимум `weed_dicot`, `weed_monocot`)
- `tests/__init__.py`, `tests/conftest.py`, `tests/test_settings.py`

## Интерфейсные контракты
```python
# config/settings.py
class NetworkSettings(BaseModel):
    host: str = "192.168.4.1"
    port: int = 8000
    request_timeout_s: float = 2.0          # <= 2.0 (НФТ)
    poll_rate_hz: float = 1.0               # диапазон 0.5..2.0
    reconnect_backoff_initial_s: float
    reconnect_backoff_max_s: float
    reconnect_backoff_factor: float
    download_chunk_bytes: int

class DetectorSettings(BaseModel):
    model_path: Path
    classes_path: Path
    confidence_threshold: float
    nms_iou_threshold: float
    input_size: tuple[int, int]
    video_frame_skip: int                   # детекция каждого N-го кадра

class MapSettings(BaseModel):
    heatmap_cell_m: float = 0.5
    critical_density_threshold: float
    max_local_extent_m: float = 3.0
    wgs84_origin: tuple[float, float] | None   # опорная точка; без неё local→wgs84 запрещена

class AppSettings(BaseSettings):
    network: NetworkSettings
    detector: DetectorSettings
    map: MapSettings
    workdir: Path
    @classmethod
    def load(cls, path: Path | None = None) -> "AppSettings": ...
    def save(self, path: Path) -> None: ...

def get_settings() -> AppSettings: ...
def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None: ...
```
`main.py`: `def main() -> int` — загрузка настроек, настройка логов, создание `QApplication` (окно подключается позже, в T-12).

## Критерии приемки (Definition of Done)
- [ ] `pip install -e .` и `pip install -r requirements.txt` проходят в чистом venv.
- [ ] Структура каталогов полностью соответствует разделу «Структура проекта» ТЗ (+ добавлен пакет `controllers/`).
- [ ] `AppSettings.load()` без файла возвращает значения по умолчанию из `default_config.json`.
- [ ] Валидация отклоняет `request_timeout_s > 2.0`, `poll_rate_hz` вне 0.5–2.0, `heatmap_cell_m <= 0`.
- [ ] В зависимостях нет компонентов под AGPL (проверено списком лицензий в `AGENTS.md`).
- [ ] `core/` и `schemas/` не импортируют PySide6 (зафиксировано в `AGENTS.md` и проверяется тестом T-19).

## План верификации / Тесты
- `tests/test_settings.py`: значения по умолчанию, round-trip `save`/`load`, негативные кейсы валидации.
- Команда `python -c "import geo_weed_station"` выполняется без ошибок.
- `pytest -q` запускается (пустые тесты зелёные), `conftest.py` содержит фикстуры `tmp_workdir`, `settings`.
