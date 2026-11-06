# TASK 18 — Упаковка (deploy) и проектная документация

## ID и название
`T-18` · Сборка PyInstaller, установочный скрипт Linux, документация (architecture, xmp_specification, user_guide).

## Зависимости (Prerequisites)
T-12, T-16, T-17.

## Целевые файлы
- `deploy/build_app.py`
- `deploy/installer_linux.sh`
- `deploy/geo_weed_station.spec` (PyInstaller spec: ресурсы QSS/иконки, `models/`, `default_config.json`)
- `docs/architecture.md` (слои, потоки, диаграммы сигналов)
- `docs/xmp_specification.md` (теги, пространства имён, примеры JPEG/TIFF)
- `docs/user_guide.md` (руководство оператора, режимы, экспорт, регламентный слот 10 минут)
- `docs/third_party_licenses.md` (аудит лицензий: MIT/Apache/BSD, отсутствие AGPL)
- `AGENTS.md` (актуализация)

## Интерфейсные контракты
```python
# deploy/build_app.py
def collect_data_files() -> list[tuple[str, str]]: ...
def build(onefile: bool = False, clean: bool = True) -> Path: ...
def smoke_test(binary: Path) -> bool: ...        # запуск с --selftest
def main(argv: Sequence[str] | None = None) -> int: ...

# main.py
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace: ...   # --config, --selftest, --offscreen
```
`installer_linux.sh`: создаёт venv, ставит `requirements.txt`, проверяет версию Python и системные Qt-зависимости, создаёт launcher. Поведение `--selftest`: загрузка настроек, импорт всех подпакетов, создание `QApplication` в offscreen, проверка наличия ресурсов, код возврата 0/1.

## Критерии приемки (Definition of Done)
- [x] `python deploy/build_app.py` формирует исполняемый артефакт; `--selftest` бинарника возвращает 0 на чистой машине без Python.
- [x] Ресурсы (QSS, иконки, `classes.json`, ONNX-модель) находятся в упакованном приложении (пути через `sys._MEIPASS`-совместимый резолвер).
- [x] `installer_linux.sh` идемпотентен и завершается успешно в чистом Ubuntu LTS.
- [x] Документы соответствуют реальной реализации; XMP-спецификация совпадает с `xmp_parser`.
- [x] Аудит лицензий не содержит AGPL (в частности, нет ultralytics/PyQt).

## План верификации / Тесты
`tests/test_deploy.py`: сборка в temp-директорию (маркер `slow`), `--selftest`, проверка состава `datas`. Ручной чек-лист: запуск бинарника, переключение режимов, открытие изображения.
