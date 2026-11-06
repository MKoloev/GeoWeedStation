# TASK 19 — Сквозные тесты, проверка НФТ и архитектурных ограничений

## ID и название
`T-19` · E2E-сценарии (офлайн и онлайн с мок-сервером), нагрузочные проверки НФТ, архитектурные тесты импортов.

## Зависимости (Prerequisites)
T-16, T-17, T-18.

## Целевые файлы
- `tests/e2e/test_offline_flow.py`
- `tests/e2e/test_online_flow.py`
- `tests/e2e/test_resilience.py`
- `tests/nfr/test_performance.py`
- `tests/architecture/test_import_rules.py`
- `tests/helpers/dataset_factory.py` (синтетический датасет: JPEG/TIFF с XMP, битые файлы, видео)
- `pytest.ini` / секция в `pyproject.toml` (маркеры `slow`, `integration`, `e2e`, `nfr`)
- `.github/workflows/ci.yml` (опционально: pytest в offscreen-режиме)
- `docs/test_report_template.md`

## Интерфейсные контракты
```python
# tests/helpers/dataset_factory.py
def make_geotagged_jpeg(path: Path, *, cs: CoordinateSystem, altitude_m: float | None, yaw_deg: float, xy: tuple[float, float]) -> Path: ...
def make_broken_jpeg(path: Path) -> Path: ...
def make_video(path: Path, frames: int, fps: int) -> Path: ...
def make_dataset(root: Path, n_valid: int, n_no_xmp: int, n_broken: int) -> Path: ...

# tests/architecture/test_import_rules.py
def collect_imports(package_dir: Path) -> dict[str, set[str]]: ...
# правила: core/ и schemas/ ↛ PySide6; network/ ↛ ui, workers; workers/ ↛ ui; ui/ ↛ network (только через controllers)
```
Сценарии E2E:
1. Офлайн: папка с `local_aruco`-кадрами → пакет → карта/heatmap → экспорт CSV/GeoJSON/PNG → сверка координат с эталоном.
2. Офлайн: кадры без XMP → карта заблокирована, экспорт недоступен.
3. Офлайн: видео с рамками, отмена обработки.
4. Онлайн: mock-сервер, connect → plan → start → live-карта → completed → скачать → удалить.
5. Устойчивость: `fail_mode=drop/timeout` → reconnect, GUI-поток не блокируется.
6. Вторичная обработка `COLLECTION_ONLY`-запуска локальным ONNX.

## Критерии приемки (Definition of Done)
- [x] Все E2E-сценарии проходят в `QT_QPA_PLATFORM=offscreen` без физического БПЛА и внешней сети.
- [x] НФТ подтверждены: XMP ≤ 50 мс; таймаут ≤ 2 с; отрисовка позы ≤ 100 мс; видео UI ≥ 30 FPS; кадр 8000×6000 без OOM.
- [x] Архитектурный тест запрещает нарушающие слои импорты и циклы.
- [x] Покрытие `core/`, `schemas/`, `network/` ≥ 85 %.
- [x] Прогон полного набора детерминирован (повтор ×3 без флаки).
- [x] Ограничение 10-минутного слота и куб 3×3×3 м проверены тестами планировщика/UI.

## План верификации / Тесты
`pytest -m "not slow"` — быстрый набор для CI; `pytest -m "e2e or nfr"` — полный. Результаты фиксируются по шаблону `docs/test_report_template.md`; открытые дефекты заводятся списком в отчёте.
