# TASK 15 — UI: диалоговые окна (настройки и экспорт)

## ID и название
`T-15` · `settings_dialog`, `export_dialog`, диалог подтверждения удаления/аварийных действий.

## Зависимости (Prerequisites)
T-01, T-06, T-12.

## Целевые файлы
- `src/geo_weed_station/ui/dialogs/settings_dialog.py`
- `src/geo_weed_station/ui/dialogs/export_dialog.py`
- `src/geo_weed_station/ui/dialogs/confirm_dialog.py`
- `tests/ui/test_settings_dialog.py`, `tests/ui/test_export_dialog.py`

## Интерфейсные контракты
```python
class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None: ...
    def result_settings(self) -> AppSettings: ...        # провалидированная копия
    # вкладки: Сеть (host, port, timeout, poll Hz), Модель (путь, пороги, frame skip),
    #          Карта (ячейка heatmap, критический порог, origin WGS84)

class ExportFormat(StrEnum): CSV; GEOJSON; PNG
class ExportRequest(BaseModel):
    formats: set[ExportFormat]
    target_dir: Path
    base_name: str
    png_options: PngExportOptions | None

class ExportDialog(QDialog):
    def __init__(self, coord_system: CoordinateSystem, has_orthophoto: bool, parent: QWidget | None = None) -> None: ...
    def result_request(self) -> ExportRequest: ...

class ConfirmDialog:
    @staticmethod
    def confirm_delete_run(run_id: str, size_bytes: int, parent: QWidget | None = None) -> bool: ...
    @staticmethod
    def confirm_abort(action: StopAction, parent: QWidget | None = None) -> bool: ...
```
Диалоги не выполняют запись файлов и не сохраняют настройки сами — только возвращают валидные DTO.

## Критерии приемки (Definition of Done)
- [x] Валидация в диалоге повторяет ограничения `AppSettings` (таймаут ≤ 2 с, частота 0.5–2 Гц и т. п.), кнопка OK блокируется при ошибках.
- [x] Экспорт: выбор любых форматов; настройки PNG (фон, сетка, легенда, масштаб, DPI) доступны только при выбранном PNG.
- [x] Вариант фона «ортофотоплан» недоступен без `has_orthophoto`.
- [x] Для `local_aruco` отображается пометка о локальной CRS в метрах; для `wgs84` — EPSG:4326.
- [x] Диалог удаления показывает объём освобождаемого места.
- [x] Отмена диалога не меняет текущие настройки.

## План верификации / Тесты
`pytest-qt`: открытие с настройками → изменение → `result_settings` отражает изменения; невалидные значения → OK заблокирован; комбинации форматов экспорта; поведение `ConfirmDialog` с замоканным `QMessageBox`.
