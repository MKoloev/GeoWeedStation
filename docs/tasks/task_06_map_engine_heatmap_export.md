# TASK 06 — MapEngine: WeedMap, тепловая карта, статистика, экспорт

## ID и название
`T-06` · Агрегация слоёв карты, Heatmap (суммирование/KDE), метрики засоренности, экспорт CSV/GeoJSON/PNG.

## Зависимости (Prerequisites)
T-02, T-05.

## Целевые файлы
- `src/geo_weed_station/core/map_engine/weed_map.py` (`WeedMap`, `MapEngine`)
- `src/geo_weed_station/core/map_engine/heatmap.py`
- `src/geo_weed_station/core/map_engine/statistics.py`
- `src/geo_weed_station/core/map_engine/exporter.py`
- `src/geo_weed_station/schemas/map_models.py` (`WeedPoint`, `HeatmapGrid`, `MapStatistics`)
- `tests/test_heatmap.py`, `tests/test_map_engine.py`, `tests/test_exporter.py`

## Интерфейсные контракты
```python
class WeedPoint(BaseModel):  weed_id; class_name; confidence; x; y; coord_system; source_frame
class HeatmapGrid(BaseModel):
    origin: tuple[float, float]; cell_size_m: float
    matrix: Any                       # np.ndarray, исключён из сериализации
class MapStatistics(BaseModel):
    total_weeds: int; by_class: dict[str, int]
    surveyed_area_m2: float; mean_density_per_m2: float; critical_area_percent: float

class HeatmapAlgorithm(ABC):          # точка расширения (KDE, Gaussian, hex)
    @abstractmethod
    def compute(self, points: Sequence[WeedPoint], cell_m: float, bounds: Bounds) -> HeatmapGrid: ...
class CountHeatmap(HeatmapAlgorithm): ...
class KdeHeatmap(HeatmapAlgorithm):
    def __init__(self, bandwidth_m: float) -> None: ...

class WeedMap:
    points: list[WeedPoint]; coord_system: CoordinateSystem
    def add_points(self, pts: Iterable[WeedPoint]) -> None: ...
    def bounds(self) -> Bounds: ...
    def convex_hull_area(self) -> float: ...

class MapEngine:
    def __init__(self, projector: ProjectionStrategy, settings: MapSettings, heatmap: HeatmapAlgorithm | None = None) -> None: ...
    def ingest_frame(self, payload: FramePayload) -> list[WeedPoint]: ...   # GeoreferenceMissingError при блокировке
    def ingest_summary(self, summary: SummaryDetections) -> list[WeedPoint]: ...  # борт уже посчитал координаты
    def build_heatmap(self, cell_m: float | None = None) -> HeatmapGrid: ...
    def statistics(self) -> MapStatistics: ...
    def is_locked(self) -> bool: ...
    def reset(self) -> None: ...

# exporter.py
class MapExporter:
    def to_csv(self, wm: WeedMap, path: Path) -> None: ...      # weed_id,class_name,confidence,coord_x,coord_y,coord_system,source_frame
    def to_geojson(self, wm: WeedMap, path: Path) -> None: ...  # FeatureCollection/Point; local → явный CRS в метаданных, wgs84 → [lon, lat] EPSG:4326
    def to_png(self, wm: WeedMap, grid: HeatmapGrid | None, path: Path, options: PngExportOptions) -> None: ...
class PngExportOptions(BaseModel):
    background: Literal["white", "transparent", "orthophoto"]; show_grid; show_legend; show_scalebar; dpi: int
    orthophoto_path: Path | None
```

## Критерии приемки (Definition of Done)
- [ ] Heatmap: сумма ячеек == числу точек (для `CountHeatmap`); размер ячейки задаётся (0.5 / 1.0 м).
- [ ] Метрики: площадь по Convex Hull, плотность на м², процент ячеек выше критического порога.
- [ ] Экспорт блокируется (`GeoreferenceMissingError`), если карта заблокирована.
- [ ] CSV содержит ровно указанные столбцы; GeoJSON валиден (RFC 7946) и содержит `properties` растения.
- [ ] PNG содержит сетку, легенду, шкалу плотности, масштабную линейку; поддерживает 3 типа фона.
- [ ] Движок не зависит от источника данных и PySide6; растровая отрисовка PNG — через OpenCV/Pillow/numpy.
- [ ] Алгоритм heatmap подменяется без изменения `MapEngine`.

## План верификации / Тесты
`test_heatmap.py` (агрегация, границы ячеек, KDE нормировка), `test_map_engine.py` (ingest frame/summary, lock, stats, reset), `test_exporter.py` (парсинг CSV/GeoJSON, размеры/режим PNG, прозрачный фон имеет alpha).
