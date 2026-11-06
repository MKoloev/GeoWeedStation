# TASK 02 — Pydantic-схемы и DTO-контракты

## ID и название
`T-02` · Полный комплект схем данных: сущности кадра, FramePayload, миссия, телеметрия, REST-DTO борта.

## Зависимости (Prerequisites)
T-01.

## Целевые файлы
- `src/geo_weed_station/schemas/entities.py`
- `src/geo_weed_station/schemas/payload.py`
- `src/geo_weed_station/schemas/mission.py`
- `src/geo_weed_station/schemas/telemetry.py`
- `src/geo_weed_station/schemas/onboard_api.py` (DTO запросов/ответов REST борта)
- `src/geo_weed_station/schemas/__init__.py` (реэкспорт)
- `tests/test_schemas.py`

## Интерфейсные контракты
Все модели: `model_config = ConfigDict(frozen=True, extra="forbid")` (защита от изменения структуры).

```python
# entities.py
class CoordinateSystem(StrEnum):
    LOCAL_ARUCO = "local_aruco"     # внутреннее имя (XMP, MapEngine)
    WGS84 = "wgs84"

class BBox(BaseModel):              # нормализованные 0..1
    x: float; y: float; w: float; h: float
    def to_xyxy(self) -> tuple[float, float, float, float]: ...

class DetectedWeed(BaseModel):
    weed_id: str
    class_id: int
    class_name: str
    confidence: float               # 0..1
    bbox: BBox
    world_x: float | None = None    # X или Lon
    world_y: float | None = None    # Y или Lat

class CameraSpatialPose(BaseModel):
    coordinate_system: CoordinateSystem
    x: float; y: float              # метры или lon/lat
    altitude_m: float               # относительная высота H
    roll_deg: float; pitch_deg: float; yaw_deg: float

class SensorOptics(BaseModel):
    focal_length_mm: float
    fov_h_deg: float | None; fov_v_deg: float | None
    sensor_width_mm: float | None; sensor_height_mm: float | None
    principal_point_px: tuple[float, float] | None
    band_names: list[str]

# payload.py
class FramePayload(BaseModel):
    frame_id: str
    source_path: str | None
    image_size_px: tuple[int, int]
    image: Any | None = Field(default=None, exclude=True)   # np.ndarray, вне сериализации
    pose: CameraSpatialPose | None
    optics: SensorOptics | None
    detections: list[DetectedWeed]
    def has_georef(self) -> bool: ...   # H > 0, координаты и углы заданы

# mission.py
class Waypoint(BaseModel):
    x: float; y: float; z: float; yaw: float
class MissionMode(StrEnum):
    COLLECTION_ONLY = "collection_only"        # use_nn=False
    INFERENCE_SEQUENTIAL = "inference_sequential"  # use_nn=True
class SavePolicy(StrEnum):
    ALL = "all"; IMAGES_ONLY = "images_only"; METADATA_ONLY = "metadata_only"
class MissionTask(BaseModel):
    coordinate_system: CoordinateSystem
    mode: MissionMode
    save_policy: SavePolicy
    waypoints: list[Waypoint]
    @model_validator(mode="after")
    def _validate_task(self) -> "MissionTask": ...   # COLLECTION_ONLY + METADATA_ONLY запрещён; waypoints не пуст

# telemetry.py
class FlightState(StrEnum):
    IDLE; TAKEOFF; FLYING_TO_WP; PROCESSING; LANDING; COMPLETED; ABORTED; ERROR
class DronePose(BaseModel):
    x; y; z; roll; pitch; yaw: float; coordinate_system: str; timestamp: float
class DroneStatus(BaseModel):
    preset: str; flight_state: FlightState
    current_wp_index: int; total_waypoints: int
    current_pose: DronePose | None
    captures_count: int; is_data_ready: bool; active_run_id: str | None
class ConnectionState(StrEnum):
    DISCONNECTED; CONNECTING; CONNECTED; RECONNECTING; TIMEOUT; ERROR

# onboard_api.py  (строго по agrovision_onboard_rest_api_specification.md)
class StartRequest(BaseModel):   # coordinate_system: "aruco_local"|"wgs84", use_nn, save_policy, waypoints
class StartResponse(BaseModel):  # status, run_id, run_path, total_waypoints
class StopAction(StrEnum): LAND = "land"; RTH = "rth"
class StopRequest(BaseModel):    # action: StopAction
class StopResponse(BaseModel):   # status, action_initiated, active_run_id, message
class DroneRun(BaseModel):       # run_id, start_time, status, directory_path, size_bytes, contents
class SummaryDetection(BaseModel):  # class_name, confidence, bbox_px[4], target_local_coords{x,y}
class SummaryItem(BaseModel):       # wp_index, timestamp, camera_x, camera_y, altitude_m, detections
class SummaryDetections(BaseModel): # run_id, coordinate_system, detections_count, items
class DeleteRunResponse(BaseModel): # deleted_run_id, freed_bytes, status

def mission_to_start_request(task: MissionTask) -> StartRequest: ...
def summary_to_frame_payloads(summary: SummaryDetections) -> list[FramePayload]: ...
```
Маппинг имён СК: внутреннее `local_aruco` ↔ сетевое `aruco_local` выполняется только в `onboard_api.py`.

## Критерии приемки (Definition of Done)
- [ ] Все модели иммутабельны, запрещены лишние поля.
- [ ] `FramePayload.has_georef()` возвращает `False` при `pose is None`, `altitude_m <= 0` или отсутствии углов.
- [ ] `MissionTask` отклоняет `COLLECTION_ONLY` + `METADATA_ONLY` (холостой полёт).
- [ ] JSON-примеры из спецификации борта (`/status`, `/runs`, `summary`, `start`) парсятся без ошибок.
- [ ] `mission_to_start_request` корректно мапит `mode→use_nn`, `local_aruco→aruco_local`.
- [ ] Модуль не импортирует PySide6, cv2, onnxruntime.

## План верификации / Тесты
- `tests/test_schemas.py`: валидные/невалидные кейсы каждой модели, парсинг эталонных JSON из спецификации (хранить в `tests/data/api_samples/*.json`), проверка `frozen`, `has_georef`, мапперов.
