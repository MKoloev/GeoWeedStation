# TASK 12 — UI-каркас: главное окно, переключатель режимов, статус-панель, стили

## ID и название
`T-12` · `main_window`, `status_bar_panel`, QSS-тема, механизм блокировки элементов по режиму.

## Зависимости (Prerequisites)
T-01, T-02, T-10 (типы `ConnectionState`, сигналы воркеров).

## Целевые файлы
- `src/geo_weed_station/ui/main_window.py`
- `src/geo_weed_station/ui/widgets/status_bar_panel.py`
- `src/geo_weed_station/ui/app_mode.py` (`AppMode`, `ModeManager`)
- `src/geo_weed_station/ui/resources/styles.qss`
- `src/geo_weed_station/ui/resources/icons/` (SVG-иконки)
- `main.py` (подключение главного окна)
- `tests/ui/test_main_window.py`, `tests/ui/test_status_bar_panel.py`

## Интерфейсные контракты
```python
class AppMode(StrEnum): ONLINE = "online"; OFFLINE = "offline"

class ModeManager(QObject):
    mode_changed = Signal(AppMode)
    @property
    def mode(self) -> AppMode: ...
    @Slot(AppMode)
    def set_mode(self, mode: AppMode) -> None: ...

class StatusBarPanel(QWidget):
    cancel_requested = Signal()
    @Slot(str)
    def set_message(self, text: str) -> None: ...
    @Slot(int)
    def set_progress(self, percent: int) -> None: ...       # 0..100
    @Slot(ConnectionState)
    def set_connection_state(self, s: ConnectionState) -> None: ...
    def set_cancel_enabled(self, enabled: bool) -> None: ...
    def set_counters(self, captures: int | None, weeds: int | None) -> None: ...

class MainWindow(QMainWindow):
    request_settings = Signal(); request_export = Signal()
    open_image_requested = Signal(); open_folder_requested = Signal(); open_video_requested = Signal()
    def add_mode_panel(self, mode: AppMode, widget: QWidget) -> None: ...
    def central_tabs(self) -> QTabWidget: ...     # Изображение / Видео / Карта / Миссия / Запуски
    def set_map_tab_locked(self, locked: bool, reason: str | None = None) -> None: ...
    def closeEvent(self, event: QCloseEvent) -> None: ...   # корректная остановка всех QThread
```
Поведение: селектор режима в верхней панели; в `OFFLINE` скрываются сетевая панель и телеметрия, в `ONLINE` — проводник локальных файлов; блокировка вкладки карты показывает «Геопривязка отсутствует. Доступен только просмотр исходных изображений»; меню и тулбар содержат только сигналы (бизнес-логики нет — MVC).

## Критерии приемки (Definition of Done)
- [ ] Окно запускается из `main.py`, тема QSS применена, иконки загружаются.
- [ ] Переключение режима скрывает/показывает ровно нужные панели; состояние сохраняется между сеансами.
- [ ] Статус-бар: текст операции, прогресс 0–100, кнопка отмены, индикатор соединения (5 состояний).
- [ ] Блокировка вкладки карты работает и отображает требуемое предупреждение.
- [ ] `closeEvent` останавливает воркеры и ждёт завершения потоков (без «QThread destroyed»).
- [ ] Модуль не содержит вызовов сети, OpenCV, ONNX.

## План верификации / Тесты
`pytest-qt` (offscreen): переключение режимов (видимость виджетов), эмиссия `cancel_requested`, границы прогресса, блокировка карты, закрытие окна при «запущенном» фиктивном воркере. Ручной чек-лист скриншотов в `docs/user_guide.md` (T-18).
