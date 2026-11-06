# TASK 07 — Планировщик миссий (галсы / «змейка»)

## ID и название
`T-07` · `mission_planner`: расчёт шага сетки и генерация маршрутных точек.

## Зависимости (Prerequisites)
T-02.

## Целевые файлы
- `src/geo_weed_station/core/mission_planner.py`
- `tests/test_mission_planner.py`

## Интерфейсные контракты
```python
class PlannerParams(BaseModel):
    corner_tl: tuple[float, float]       # (X1, Y1)
    corner_br: tuple[float, float]       # (X2, Y2)
    coordinate_system: CoordinateSystem
    altitude_m: float                    # Z
    optics: SensorOptics
    overlap: float = 0.1                 # 0..0.9
    mode: MissionMode
    save_policy: SavePolicy
    step_override_m: float | None = None
    start_yaw_rad: float = 0.0

def compute_footprint_m(optics: SensorOptics, altitude_m: float) -> tuple[float, float]: ...
def compute_grid_step(optics: SensorOptics, altitude_m: float, overlap: float) -> tuple[float, float]: ...
def generate_waypoints(params: PlannerParams) -> list[Waypoint]: ...   # серпантин, упорядочено
def build_mission(params: PlannerParams) -> MissionTask: ...
def validate_params(params: PlannerParams, max_extent_m: float = 3.0) -> list[str]: ...  # список ошибок; пусто = OK
def estimate_flight_time_s(waypoints: Sequence[Waypoint], speed_mps: float, hover_s: float) -> float: ...
```
Правила: для `local_aruco` область и точки обязаны лежать в кубе 3×3×3 м; для `wgs84` ограничения куба не применяются; `COLLECTION_ONLY` без сохранения изображений отклоняется; оценка времени предупреждает о превышении слота 10 мин.

## Критерии приемки (Definition of Done)
- [ ] Порядок точек — «змейка»: чётные ряды слева-направо, нечётные справа-налево; нет дубликатов.
- [ ] Все точки внутри заданного прямоугольника (с учётом края), `z` = заданной высоте.
- [ ] Шаг вычисляется из FOV/фокуса и перекрытия; `step_override_m` имеет приоритет.
- [ ] Валидация возвращает понятные сообщения (вырожденный прямоугольник, H ≤ 0, выход из куба, конфликт save_policy).
- [ ] Результат `build_mission` проходит валидацию `MissionTask` (T-02).
- [ ] Модуль чистый: без PySide6, сети, файлового ввода-вывода.

## План верификации / Тесты
`tests/test_mission_planner.py`: сетка 3×3 м при H=1.5 м (количество точек и порядок), перевёрнутые углы, одна строка/один столбец, границы куба, оценка времени, параметризованный `overlap`.
