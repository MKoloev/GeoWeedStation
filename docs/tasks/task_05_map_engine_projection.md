# TASK 05 — MapEngine: проецирование и системы координат

## ID и название
`T-05` · Стратегии проецирования пикселей в плоскость поля, конвертация `local_aruco` ↔ `wgs84`, блокировка при отсутствии геопривязки.

## Зависимости (Prerequisites)
T-01, T-02.

## Целевые файлы
- `src/geo_weed_station/core/map_engine/__init__.py`
- `src/geo_weed_station/core/map_engine/projector.py`
- `src/geo_weed_station/core/map_engine/coordinates.py`
- `src/geo_weed_station/core/map_engine/errors.py` (`GeoreferenceMissingError`, `OriginNotSetError`)
- `tests/test_projector.py`
- `tests/test_coordinates.py`

## Интерфейсные контракты
```python
class ProjectionStrategy(ABC):                       # паттерн Strategy
    @abstractmethod
    def project(self, payload: FramePayload) -> list[DetectedWeed]: ...

class NadirProjector(ProjectionStrategy):
    def __init__(self, distortion: DistortionModel | None = None) -> None: ...
    def project(self, payload: FramePayload) -> list[DetectedWeed]: ...
    @staticmethod
    def ground_footprint_m(optics: SensorOptics, altitude_m: float, image_size_px: tuple[int, int]) -> tuple[float, float]: ...

class DistortionModel(Protocol):                     # точка расширения
    def undistort_point(self, x_px: float, y_px: float) -> tuple[float, float]: ...

def validate_georef(payload: FramePayload) -> None: ...   # raises GeoreferenceMissingError; H>0, координаты, углы
def get_projector(name: str = "nadir") -> ProjectionStrategy: ...   # фабрика

# coordinates.py
class CoordinateConverter:
    def __init__(self, wgs84_origin: tuple[float, float] | None) -> None: ...
    def local_to_wgs84(self, x: float, y: float) -> tuple[float, float]: ...   # OriginNotSetError без origin
    def wgs84_to_local(self, lat: float, lon: float) -> tuple[float, float]: ...
    @staticmethod
    def rotate_by_yaw(dx: float, dy: float, yaw_deg: float) -> tuple[float, float]: ...
```
Алгоритм: нормализованный центр бокса → смещение от главной точки → метры через размер проекции кадра (надир, `Pitch≈-90°`) → поворот на Yaw → сложение с координатами камеры. Неявная конвертация `local_aruco → wgs84` запрещена.

## Критерии приемки (Definition of Done)
- [ ] `H <= 0`, отсутствие pose/углов/центра → `GeoreferenceMissingError`, проекции не выполняются.
- [ ] Для Yaw = 0 и центра кадра результат = координаты камеры; для угловых боксов совпадает с аналитикой.
- [ ] Поворот Yaw корректен для 0°, 90°, ±180°.
- [ ] Результат проекции записывается в `DetectedWeed.world_x/world_y`; исходные объекты не мутируются.
- [ ] `local_to_wgs84` без origin → `OriginNotSetError`; round-trip ошибка < 1 мм (локальная) / < 1e-8° .
- [ ] Модуль не знает об источнике данных (XMP/REST) и не импортирует PySide6/cv2.
- [ ] Новая стратегия подключается без правки существующего кода (регистр стратегий).

## План верификации / Тесты
`tests/test_projector.py`: параметризованные кейсы (H, f, sensor, yaw), граничные значения, blocked-случаи. `tests/test_coordinates.py`: round-trip, отсутствие origin, rotate_by_yaw. Допуск точности — 1 см на кадре 2×2 м.
