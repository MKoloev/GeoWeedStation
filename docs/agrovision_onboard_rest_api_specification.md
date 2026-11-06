# Спецификация REST API AgrovisionOnboard

Документ описывает сетевой протокол взаимодействия внешних приложений (включая наземную станцию оператора **GeoWeedStation**, скрипты автоматизации и внешние ГИС) с бортовым программным модулем **agrovision_onboard**.

---

## 1. Общие сведения и сетевой протокол

- **Базовый стек:** HTTP/1.1, REST, JSON (Pydantic-сериализация).
- **Сетевой транспорт:** TCP/IP (Wi-Fi / Ethernet / радиомодем).
- **Порт по умолчанию:** `http://<drone_ip>:8000` (настраивается в `settings.py`).
- **Формат данных запросов и ответов:** `application/json; charset=utf-8` (за исключением бинарной выгрузки архивов и снимков).
- **Коды состояний HTTP:**
  - `200 OK`: Успешное синхронное выполнение или получение данных.
  - `202 Accepted`: Миссия принята, запущена процедура взлета/полета в фоновом режиме.
  - `400 Bad Request`: Ошибка валидации параметров (несовместимый пресет, конфликт каналов камеры и требований нейросети).
  - `404 Not Found`: Запрашиваемый запуск (`run_id`) или файл не найден.
  - `409 Conflict`: Попытка запуска миссии при уже выполняющемся полете, либо удаление данных активной миссии.
  - `500 Internal Server Error`: Аппаратный сбой периферии или автопилота.

---

## 2. Жизненный цикл миссии (State Machine)

Внешнее приложение должно контролировать процесс облета через опрос статуса (`GET /status`):

```text
[ IDLE ]  ---> POST /start ---> [ TAKEOFF ] ---> [ FLYING_TO_WP ]
    ^                                                    |
    |                                            [ CAPTURE / INFERENCE ]
    |                                                    |
[ FINISHED / ERROR ] <--- [ LANDING ] <------------------+ (все точки пройдены
                                                            или POST /stop)
```

---

## 3. Спецификация точек входа (Endpoints)

### 3.1. Запуск полетного задания (`POST /start`)

Инициирует создание директории запуска на борту, валидирует конфигурацию пресета и запускает автономный цикл миссии.

- **URL:** `/start`
- **Метод:** `POST`
- **Заголовки:** `Content-Type: application/json`

#### Тело запроса (JSON):
```json
{
  "coordinate_system": "aruco_local",
  "use_nn": true,
  "save_policy": "all",
  "waypoints": [
    { "x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0 },
    { "x": 1.2, "y": 0.0, "z": 1.5, "yaw": 0.0 },
    { "x": 1.2, "y": 1.2, "z": 1.5, "yaw": 1.57 }
  ]
}
```

*Параметры:*
- `coordinate_system` (string, enum): `"aruco_local"` (декартова в метрах) или `"wgs84"` (географическая).
- `use_nn` (boolean): `true` — последовательный инференс на борту (`INFERENCE_SEQUENTIAL`), `false` — только сбор данных (`COLLECTION_ONLY`).
- `save_policy` (string, enum): `"all"` (снимки + детекции), `"images_only"`, `"metadata_only"`.
- `waypoints` (array): Упорядоченный список целевых точек маршрута.

#### Пример ответа (`202 Accepted`):
```json
{
  "status": "started",
  "run_id": "run_20261105_153022",
  "run_path": "/home/pi/agrovision_onboard/runs/run_20261105_153022",
  "total_waypoints": 3
}
```

---

### 3.2. Экстренное / штатное прерывание (`POST /stop`)

Немедленно останавливает облет маршрута, закрывает текущие дескрипторы записи и сажает дрон.

- **URL:** `/stop`
- **Метод:** `POST`
- **Заголовки:** `Content-Type: application/json`

#### Тело запроса (JSON):
```json
{
  "action": "land"
}
```

*Параметры:*
- `action` (string, enum): `"land"` (посадка на текущем месте) или `"rth"` (возврат в точку старта и посадка).

#### Пример ответа (`200 OK`):
```json
{
  "status": "aborting",
  "action_initiated": "land",
  "active_run_id": "run_20261105_153022",
  "message": "Mission aborted by operator. Executing landing procedure."
}
```

---

### 3.3. Мониторинг состояния бортовой подсистемы (`GET /status`)

Рекомендуемая частота опроса наземной станцией: 1–5 Гц.

- **URL:** `/status`
- **Метод:** `GET`

#### Пример ответа (`200 OK`):
```json
{
  "preset": "clever4_csi_onnx",
  "flight_state": "flying_to_wp",
  "current_wp_index": 2,
  "total_waypoints": 3,
  "current_pose": {
    "x": 1.18,
    "y": 0.45,
    "z": 1.51,
    "roll": 0.01,
    "pitch": -0.02,
    "yaw": 0.05,
    "coordinate_system": "aruco_local",
    "timestamp": 1793892622.412
  },
  "captures_count": 1,
  "is_data_ready": false,
  "active_run_id": "run_20261105_153022"
}
```

*Возможные значения `flight_state`:* `"idle"`, `"takeoff"`, `"flying_to_wp"`, `"processing"`, `"landing"`, `"completed"`, `"aborted"`, `"error"`.

---

### 3.4. Получение списка завершенных и текущих миссий (`GET /runs`)

Сканирует бортовую директорию `runs/` и возвращает манифесты полетов.

- **URL:** `/runs`
- **Метод:** `GET`

#### Пример ответа (`200 OK`):
```json
[
  {
    "run_id": "run_20261105_153022",
    "start_time": "2026-11-05T15:30:22Z",
    "status": "completed",
    "directory_path": "/home/pi/agrovision_onboard/runs/run_20261105_153022",
    "size_bytes": 48234496,
    "contents": ["images", "detections_json", "manifest"]
  }
]
```

---

### 3.5. Выгрузка данных миссии (`GET /runs/{id}/data`)

Позволяет наземной станции скачивать результаты для последующей локальной обработки и построения карты.

- **URL:** `/runs/{id}/data`
- **Метод:** `GET`
- **Параметры запроса (Query Params):**
  - `type` (string, optional, default: `"summary"`):
    - `summary`: Возвращает единый агрегированный JSON со всеми обнаруженными зонами сорняков и координатами.
    - `archive`: Скачивает zip-архив всей директории запуска (снимки, манифест, JSON).
    - `image`: Скачивание отдельного снимка (требует параметр `wp_index`).
  - `wp_index` (integer, optional): Индекс точки маршрута (при `type=image`).

#### Пример выгрузки сводки (`type=summary`):
```json
{
  "run_id": "run_20261105_153022",
  "coordinate_system": "aruco_local",
  "detections_count": 4,
  "items": [
    {
      "wp_index": 1,
      "timestamp": 1793892625.105,
      "camera_x": 1.20,
      "camera_y": 0.00,
      "altitude_m": 1.50,
      "detections": [
        {
          "class_name": "weed_dicot",
          "confidence": 0.89,
          "bbox_px": [120, 204, 180, 260],
          "target_local_coords": { "x": 1.34, "y": 0.12 }
        }
      ]
    }
  ]
}
```

---

### 3.6. Очистка хранилища борта (`DELETE /runs/{id}`)

Удаляет данные миссии с карты памяти или диска одноплатника для предотвращения переполнения накопителя.

- **URL:** `/runs/{id}`
- **Метод:** `DELETE`

#### Пример ответа (`200 OK`):
```json
{
  "deleted_run_id": "run_20261105_153022",
  "freed_bytes": 48234496,
  "status": "success"
}
```

---

## 4. Клиентский сценарий взаимодействия (GeoWeedStation)

1. **Инициализация:** Наземная станция опрашивает `GET /status`. Убеждается, что `flight_state == "idle"`.
2. **Отправка миссии:** Формируется массив точек и отправляется `POST /start`. Станция сохраняет полученный `run_id`.
3. **Мониторинг:** В цикле вызывается `GET /status` (раз в 1 сек). Интерфейс отображает положение дрона и количество зафиксированных снимков (`captures_count`).
4. **Завершение и экспорт:** Как только `is_data_ready == true`, станция отправляет `GET /runs/{id}/data?type=summary` (или забирает сжатые кадры), выполняет проекцию координат на интерактивную карту и формирует итоговый агрономический отчет.