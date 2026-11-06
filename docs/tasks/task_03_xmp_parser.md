# TASK 03 — Парсер XMP-метаданных (JPEG/TIFF)

## ID и название
`T-03` · `xmp_parser`: извлечение и валидация пространственных метаданных из бинарников изображений.

## Зависимости (Prerequisites)
T-01, T-02.

## Целевые файлы
- `src/geo_weed_station/core/xmp_parser.py`
- `src/geo_weed_station/core/exceptions.py` (`XmpError`, `XmpNotFoundError`, `XmpMalformedError`)
- `tests/test_xmp_parser.py`
- `tests/data/xmp/` (генерируемые фикстуры; генератор в `tests/helpers/make_xmp_image.py`)

## Интерфейсные контракты
```python
@dataclass(frozen=True)
class XmpSpatialData:
    pose: CameraSpatialPose | None
    optics: SensorOptics | None
    raw_tags: dict[str, str]

def read_xmp_packet(path: Path) -> str: ...
    # JPEG: сканирование APP1 маркера; TIFF: тег 700 (XMP); без cv2/PIL-декода пикселей
def parse_xmp(xmp_xml: str) -> XmpSpatialData: ...
def extract_spatial_data(path: Path) -> XmpSpatialData: ...
def has_xmp(path: Path) -> bool: ...                  # быстрая проверка для списка файлов
def build_frame_payload(path: Path, image_size_px: tuple[int, int]) -> FramePayload: ...
    # при ошибке XMP возвращает payload с pose=None (помечается «без метаданных»)
```
Парсимые теги: `drone-dji:RelativeAltitude`, `GimbalYawDegree`, `GimbalPitchDegree`, `GimbalRollDegree`, `GpsLatitude`, `GpsLongitude`; `Camera:PerspectiveFocalLength`, `PrincipalPoint` («cx,cy»), `BandName`; `agro:CoordinateSystem`, `agro:CameraX`, `agro:CameraY`.
Правила: `local_aruco` → X/Y из `agro:*`; `wgs84` → широта/долгота с проверкой диапазонов (−90..90, −180..180). Парсер не зависит от OpenCV и НЕ читает sidecar-файлы.

## Критерии приемки (Definition of Done)
- [ ] Корректно читает XMP из JPEG (APP1) и TIFF; поддерживает атрибутную и элементную формы `rdf:Description`.
- [ ] Время извлечения одного файла ≤ 50 мс (замер в тесте на файле 8000×6000).
- [ ] Не декодирует пиксели и не загружает файл целиком в память без необходимости.
- [ ] Повреждённый/отсутствующий XMP → типизированное исключение; `build_frame_payload` не падает.
- [ ] Отсутствие обязательных тегов (высота, углы, координаты) → `pose=None` либо `has_georef()==False`.
- [ ] Для `wgs84` с невалидными полушариями/диапазонами — ошибка валидации.
- [ ] Нет обращений к внешним `.txt/.xml/.json` файлам.

## План верификации / Тесты
`tests/test_xmp_parser.py`: (1) эталонный блок из ТЗ → ожидаемые значения; (2) `local_aruco` и `wgs84`; (3) JPEG без XMP; (4) битый XML; (5) обрезанный файл; (6) TIFF; (7) бенчмарк ≤ 50 мс; (8) проверка, что при запуске не открываются файлы, кроме целевого (monkeypatch `open`).
