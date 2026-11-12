# GeoWeedStation

**GeoWeedStation** — кроссплатформенная наземная станция управления (GCS) и агрономического картографирования сорной растительности в рамках кейса **№ 2 «Геосаэро» (Автономный поиск сорняков)** по компетенции «Летающая робототехника».

Программный комплекс предназначен для планирования полетных миссий, прямого взаимодействия с бортовой системой БПЛА (**SKYRIS Техник 6S** / Orange Pi 5 Pro + Rockchip RK3588S NPU), приема телеметрии и живого видеопотока детекции, а также для локального и потокового анализа RGB и мультиспектральных аэрофотоснимков с классификацией сорняков, пространственной дедупликацией и построением цифровых карт засоренности поля.

---

## 📌 Основные возможности

* **Два режима работы**:
  * **Офлайн**: пакетная обработка аэрофотоснимков (`JPG`/`JPEG`/`PNG`/`TIFF`, включая 5-канальный мультиспектр `MS5`) с извлечением телеметрии из XMP-тегов и оптики из EXIF SubIFD, локальная детекция и классификация через ONNX Runtime, воспроизведение видеопотока с наложением детекций.
  * **Онлайн**: подключение к бортовому REST API и прямое управление бортом **Skyris (`10.42.0.1`)** по SSH/SFTP/HTTP — планирование маршрута «змейкой» в локальной системе координат (`local_aruco` / `aruco_local`), мониторинг телеметрии 1–2 Гц, просмотр видеопотока детекции с борта (`/yolo/debug`), экстренная посадка / `RTH` / `Land`, скачивание и локальная обработка собранных бортом запусков.
* **Двухстадийный ИИ-пайплайн**:
  * **Стадия 1 (Детекция)**: поддержка моделей **YOLO** и мультистрайдовых **YOLOX-M** (`d01_rgb_plant` для RGB, `d04_ms5_plant` для 5-канальных кубов `[R, G, B, RE, NIR]`), поклассовый NMS, интерактивные пороги `Confidence` и `NMS IoU`.
  * **Стадия 2 (Классификация видов и стадий)**: двухголовый классификатор **C07** (`resnet50_dualhead_c07.onnx`: 11 хозяйственных категорий + 5 фенологических стадий) и видовой классификатор **C06** (`resnet50_c06.onnx`: 24 ботанических вида `S000..S023`), автоматическая фильтрация фонового класса `background`.
* **Картографический движок (MapEngine) и аналитика**:
  * Проекция пиксельных координат сорняков на подстилающую поверхность с учетом углов камеры (Roll, Pitch, Yaw), высоты ($H > 0$), FOV и локальной метрической проекции из WGS84.
  * Пространственная дедупликация перекрывающихся кадров (`KDTree`, радиус 0.35 м) — исключение повторного подсчета одних и тех же растений.
  * Построение тепловой карты плотности (Heatmap), зон предписаний, контуров кадров и трека маршрута (старт/финиш).
  * Экспорт результатов в **CSV**, **GeoJSON**, растровую карту **PNG** (с сеткой, легендой и масштабом) и структурированный отчет (**HTML/JSON/CSV/ZIP** с замером времени обработки).
  * Защита от некорректных данных: автоматическая блокировка картографирования и экспорта с предупреждением оператора при отсутствии валидной геопривязки.
* **Высокая производительность и отзывчивость**:
  * Неблокирующий интерфейс на **PySide6** (все тяжелые операции, сеть, SSH/SFTP, инференс и декодирование выполняются в фоновых потоках `QThread`).
  * Защита от нехватки памяти (OOM) при просмотре сверхтяжелых снимков (до `8000×6000`).
* **Лицензионная чистота**:
  * Полное отсутствие компонентов под лицензиями семейства AGPL/GPL в ядре. Используются пермиссивные лицензии (MIT, Apache 2.0, BSD) и LGPLv3 для PySide6 (подробнее — [docs/third_party_licenses.md](docs/third_party_licenses.md)).

---

## 📅 Что сделано по дням и модулям (A, B, C)

Разработка велась итеративно по слоистой архитектуре (от ядра и контрактов данных к UI, интеграции с БПЛА, обучению нейросетей и испытаниям). Все 21 задача проектного плана выполнены ([docs/tasks/README.md](docs/tasks/README.md)).

### Подготовительный этап — Архитектурная база и ядро (6–9 ноября)
* Заложена слоистая архитектура (Clean Architecture / MVC), конфигурация Pydantic v2 и контракты DTO: [src/geo_weed_station/config/](src/geo_weed_station/config/), [src/geo_weed_station/schemas/](src/geo_weed_station/schemas/), [docs/architecture.md](docs/architecture.md).
* Реализованы парсинг XMP и чтение параметров оптики из EXIF SubIFD, а также локальная метрическая проекция для координат WGS84 и `local_aruco`: [src/geo_weed_station/core/xmp_parser.py](src/geo_weed_station/core/xmp_parser.py), [src/geo_weed_station/core/map_engine/](src/geo_weed_station/core/map_engine/).
* Реализован ONNX-детектор с поддержкой декодирования мультистрайдовых выходов **YOLOX**, поклассового NMS и фильтрации по порогам уверенности и IoU: [src/geo_weed_station/core/onnx_detector.py](src/geo_weed_station/core/onnx_detector.py).
* Подняты фоновые воркеры (`QThread`), сетевой REST-клиент с exponential backoff, мок-сервер бортового комплекса для автономного тестирования, UI-каркас (вкладки «Изображение», «Видео», «Карта», «Миссия», «Запуски»), контроллеры и скрипты автонастройки окружения: [src/geo_weed_station/network/](src/geo_weed_station/network/), [src/geo_weed_station/workers/](src/geo_weed_station/workers/), [src/geo_weed_station/ui/](src/geo_weed_station/ui/), [src/geo_weed_station/controllers/](src/geo_weed_station/controllers/), [tests/mocks/](tests/mocks/), [deploy/](deploy/).

### Модуль A (День C1, 10 ноября) — Проектирование, интеграция с бортом и документация
* **Проектная документация и отчет по Модулю A**: подготовлены краткое и расширенное техописание, обоснование актуальности, сравнение аналогов, распределение ролей команды, перечень оборудования и компонентов, чек-лист готовности прототипа, инструкции по развертыванию, безопасности и эксплуатации — [reports/Отчет модуль А.pdf](reports/Отчет%20модуль%20А.pdf), [reports/Отчет модуль А.docx](reports/Отчет%20модуль%20А.docx), [docs/](docs/).
* **Прямая интеграция с дроном Skyris (`10.42.0.1`)**:
  * Реализован запуск автономной миссии на борту по SSH (`paramiko`) с потоковым выводом логов, отображением прогресса и экстренной посадкой по `SIGINT`: [src/geo_weed_station/workers/ssh_mission_worker.py](src/geo_weed_station/workers/ssh_mission_worker.py), [docs/drone_communication_and_launch_plan.md](docs/drone_communication_and_launch_plan.md).
  * Добавлен просмотр живого видеопотока детекции с бортового `web_video_server:8080` (топики `/yolo/debug`, `/collect_onboard_image/debug`): [src/geo_weed_station/workers/live_stream_worker.py](src/geo_weed_station/workers/live_stream_worker.py), [src/geo_weed_station/ui/widgets/live_stream_widget.py](src/geo_weed_station/ui/widgets/live_stream_widget.py).
  * Модернизирована вкладка «Миссия» (разделение на подокна и единая автосъемка 1 Гц): [src/geo_weed_station/ui/widgets/mission_control.py](src/geo_weed_station/ui/widgets/mission_control.py).
* **Работа с разметкой и картой**: добавлены сохранение и автозагрузка разметки ([src/geo_weed_station/core/annotation_io.py](src/geo_weed_station/core/annotation_io.py)), отрисовка контуров кадров на карте и доработаны скрипты запуска для Linux и Windows.

### Модуль B (День C2, 11–12 ноября) — ИИ-пайплайн, мультиспектр, отчетность и испытания
* **Вторая стадия классификации сорняков (`T-20`)**: интегрированы ONNX-классификаторы **C07** (виды/категории + фазы развития) и **C06** (24 ботанических вида), добавлена фильтрация классов на карте и в отчетах и отсев класса `background`: [docs/tasks/task_20_weed_classification_stage.md](docs/tasks/task_20_weed_classification_stage.md), [src/geo_weed_station/ui/widgets/detection_filter_panel.py](src/geo_weed_station/ui/widgets/detection_filter_panel.py), [tests/test_onnx_classification.py](tests/test_onnx_classification.py).
* **Интеграция 5-канальной мультиспектральной модели `D04 MS5`**: реализовано чтение 5-канальных TIFF (`[R, G, B, RE, NIR]`) с предпросмотром RGB в отдельном потоке и раздельным прогоном детекции на 5 каналах + классификации на RGB-кропах: [docs/d04_ms5_plant_integration.md](docs/d04_ms5_plant_integration.md), [src/geo_weed_station/core/image_io.py](src/geo_weed_station/core/image_io.py), [src/geo_weed_station/workers/ms5_preview_worker.py](src/geo_weed_station/workers/ms5_preview_worker.py), [tests/test_ms5_integration.py](tests/test_ms5_integration.py).
* **Полный цикл приема данных с борта и отчетности**:
  * Реализованы прямой опрос видеопотока в обход системного прокси, прием геопривязки и `summary.json`, а также скачивание итоговых снимков по SFTP с прогресс-баром.
  * Реализован генератор структурированных отчетов по папке с подсчетом времени обработки, статистикой по видам/плотности, отрисовкой траектории маршрута (старт/финиш) на PNG и выгрузкой в подкаталог `report`: [src/geo_weed_station/core/report_generator.py](src/geo_weed_station/core/report_generator.py), [src/geo_weed_station/core/map_engine/exporter.py](src/geo_weed_station/core/map_engine/exporter.py).
* **Обучение и оценка моделей**: опубликован автономный комплекс обучения, экспорта (`PyTorch → ONNX FP32 → RKNN INT8` под RK3588) и расчета метрик на датасете GeosAero: [train/standalone_training/](train/standalone_training/), [train/target_model_detection_RGB/](train/target_model_detection_RGB/).
* **Испытания и верификация**: подготовлены программа и методика испытаний, проведены тесты по 10 требованиям кейса № 2 и сквозной сценарий отказоустойчивости, сформирован отчет: [reports/Modul_Б_.docx](reports/Modul_Б_.docx), [scripts/](scripts/), [tests/](tests/).

### Модуль C (День C3, 12 ноября) — Расчет экономической эффективности
* Разработана экономическая модель внедрения программно-аппаратного комплекса (базовый объем 10 000 га/сезон, горизонт 3 года, ставка дисконтирования 15%): [reports/Отчет модуль C.pdf](reports/Отчет%20модуль%20C.pdf).
* Выполнены: декомпозиция бизнес-процесса (IDEF0 A0 и A1–A4), детальная смета `CAPEX/OPEX`, расчет показателей `Payback Period`, `ROI`, `NPV`, `IRR`, сравнение 3 альтернативных вариантов реализации, матрица рисков по ISO 31010:2009 и анализ чувствительности по 3 сценариям.

---

## 🏆 Что получилось (Итоговые результаты)

1. **Программно-аппаратный комплекс полного цикла для Кейса № 2 «Геосаэро»**:
   * Выполняет автономный полет по заданному маршруту змейкой (бортовой контур SKYRIS Техник 6S + PX4 + Orange Pi 5 Pro) с трансляцией видеопотока детекции и телеметрии в станцию.
   * Поддерживает пакетную и потоковую обработку как стандартных **RGB (`JPG`/`TIFF`)**, так и **5-канальных мультиспектральных (`MS5 TIFF`)** снимков.
   * Автоматически исключает повторный учет одних и тех же растений на перекрывающихся снимках за счет пространственной кластеризации `KDTree` ([src/geo_weed_station/core/map_engine/deduplication.py](src/geo_weed_station/core/map_engine/deduplication.py)).
   * Формирует цифровую карту засоренности, датасет детекций в **JSON**, табличный экспорт **CSV**, векторные зоны-предписания **GeoJSON** для опрыскивателей, растровую карту **PNG** (300 DPI, сетка, легенда, маршрут) и структурированный отчет `report.html` с замером времени обработки.
2. **Семейство обученных моделей и автономный ML-конвейер ([train/](train/))**:
   * Подготовлен единый унифицированный датасет (`master_dataset.json`, 14 источников) и 10 конфигураций моделей (`D01–D06` детекторы YOLOX-M, `C06–C09` классификаторы ResNet-50 / ResNet-50 DualHead) с экспортом в `ONNX` (для ПК) и `RKNN INT8` (для бортового NPU Rockchip RK3588S, ~18 мс/кадр).
   * Подробные замеры качества и методология воспроизведения приведены в [train/standalone_training/README.md](train/standalone_training/README.md), [train/standalone_training/GEOSAERO_METRICS.md](train/standalone_training/GEOSAERO_METRICS.md) и [train/target_model_detection_RGB/readme.md](train/target_model_detection_RGB/readme.md).
3. **Подтвержденная экономическая эффективность ([reports/Отчет модуль C.pdf](reports/Отчет%20модуль%20C.pdf))**:
   * По расчетной модели (10 000 га/сезон, `CAPEX` 720 000 руб., `OPEX` 700 000 руб./год): прямая экономия гербицидов за счет точечного внесения — **30%** (совокупная годовая выгода с учетом сохраненного урожая — **8,5 млн руб./год**, чистый годовой поток — **7,8 млн руб.**), простой срок окупаемости — **0,09 года**, `NPV` за 3 года — **+17 089 156 руб.**
4. **Архитектурная надежность и покрытие тестами ([tests/](tests/))**:
   * Строгая изоляция чистого ядра (`core/`, `schemas/`) от GUI подтверждена AST-анализом импортов (`tests/architecture/test_import_rules.py`).
   * Написан комплексный набор модульных, UI, E2E (с мок-сервером борта) и NFR-тестов, а также тестов интеграции `C07` и `D04 MS5`.

---

## 📂 Структура проекта

```text
GeoWeedStation/
├── main.py                              # Главная точка входа приложения
├── pyproject.toml                       # Метаданные сборщика, зависимости, настройки pytest/ruff/mypy
├── requirements.txt                     # Список зависимостей Python
├── setup.sh / setup.bat                 # Скрипты автоматической установки окружения
├── run_geoweed.sh / geoweedstation.sh   # Лаунчер запуска приложения для Linux
├── run_geoweed.bat / run_geoweed_debug.bat # Лаунчеры запуска для Windows
├── AGENTS.md                            # Архитектурные правила и соглашения команды
├── LICENSE                              # Лицензия MIT
├── docs/                                # Документация проекта и спецификации
│   ├── architecture.md                  # Архитектурное описание слоев и потоков данных
│   ├── technical specification.md       # Исходное техническое задание
│   ├── agrovision_onboard_rest_api_specification.md # Спецификация бортового REST API
│   ├── drone_communication_and_launch_plan.md       # Архитектура связи и запуска дрона Skyris
│   ├── d04_ms5_plant_integration.md     # Спецификация интеграции 5-канальной модели D04 MS5
│   ├── xmp_specification.md             # Описание формата метаданных XMP
│   ├── user_guide.md                    # Руководство оператора
│   ├── third_party_licenses.md          # Аудит лицензий зависимостей
│   ├── test_report_template.md          # Шаблон отчета о тестировании
│   └── tasks/                           # Декомпозиция задач и Roadmap (Задачи 01–21)
│       └── README.md                    # Сводный Roadmap разработки
├── reports/                             # Отчетные материалы по модулям чемпионата
│   ├── Отчет модуль А.pdf / .docx       # Отчет по Модулю A (проектирование и разработка)
│   ├── Modul_Б_.docx                    # Отчет по Модулю B (ПМИ и результаты испытаний)
│   └── Отчет модуль C.pdf               # Отчет по Модулю C (экономическая эффективность)
├── src/geo_weed_station/                # Исходный код приложения
│   ├── config/                          # Настройки (AppSettings, default_config.json) и логирование
│   ├── schemas/                         # Контракты данных и Pydantic-схемы (DTO)
│   ├── core/                            # Чистое ядро: XMP, ONNX-детектор, image_io, report_generator, планировщик
│   │   └── map_engine/                  # Картография: проектор, дедупликация, heatmap, зоны, экспорт
│   ├── network/                         # HTTP REST-клиент, backoff и менеджер загрузок
│   ├── workers/                         # Фоновые потоки QThread (пакет, видео, SSH, live-поток, MS5, SFTP)
│   ├── ui/                              # Представление: виджеты, плеер, карта, фильтры, диалоги
│   └── controllers/                     # Контроллеры MVC / координация потоков и UI
├── train/                               # Код обучения, экспорта (ONNX/RKNN) и валидации нейросетей
│   ├── standalone_training/             # Автономный пакет обучения (10 моделей + master_dataset.json)
│   └── target_model_detection_RGB/      # Автономный комплект RGB-детектора (PT / ONNX / RKNN)
├── scripts/                             # Вспомогательные скрипты подготовки данных и проверки связи с БПЛА
├── tests/                               # Набор тестов (unit, UI, integration, E2E, NFR, AST)
│   ├── architecture/                    # Проверка архитектурных правил импортов (AST)
│   ├── e2e/                             # Сквозные сценарии (офлайн, онлайн, сбои)
│   ├── nfr/                             # Проверка нефункциональных требований (производительность)
│   ├── mocks/                           # Мок-сервер борта БПЛА
│   └── helpers/                         # Генераторы синтетических датасетов и XMP-снимков
├── models/                              # Модели нейросетей (.onnx и метаданные .json, скачиваются отдельно)
└── deploy/                              # Скрипты установки, упаковки и сборки (PyInstaller, Bash, Bat)
```

---

## 🚀 Установка, перенос моделей и запуск

### Системные требования
* **Python**: `3.10+`
* **ОС**: Linux (Ubuntu 20.04 / 22.04 / 24.04+) или Windows 10 / 11
* На Windows для работы `PySide6` и `onnxruntime` при необходимости установите [Microsoft Visual C++ Redistributable 2015–2022 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

---

### Шаг 1. Клонирование репозитория

```bash
git clone https://github.com/MKoloev/GeoWeedStation.git
cd GeoWeedStation
```

---

### Шаг 2. Загрузка нейронной сети (моделей)

Веса нейронной сети не хранятся в репозитории (каталог `models/` исключен в [.gitignore](.gitignore)) и скачиваются отдельно. Перед первым запуском (или сборкой дистрибутива):

#### Основной способ — готовый комплект моделей для GeoWeedStation (рекомендуется)
1. Скачайте **все** файлы нейронной сети с Google Диска:  
   https://drive.google.com/drive/folders/1L23Lig0j4bOs0tEtCuv0fgJQIMUxQpPW?usp=sharing
   *(Через браузер: «Скачать всё» и распаковать; либо через консоль:)*
   ```bash
   pip install gdown
   gdown --folder "https://drive.google.com/drive/folders/1L23Lig0j4bOs0tEtCuv0fgJQIMUxQpPW?usp=sharing" -O ./models
   ```
2. Перенесите **все скачанные файлы** в папку `models` в корне репозитория (`GeoWeedStation/models/`).

#### Полный архив всех обученных моделей проекта (включая неиспользуемые станцией)
🔗 **Ссылка на облако со всеми обученными моделями проекта:**  
https://drive.google.com/drive/folders/1QdofGN1HslQm8YPhSrMIN0_YtutROklu?usp=sharing

В этой папке собраны **все** модели, обученные в рамках проекта (включая экспериментальные и те, что напрямую не используются наземной станцией `GeoWeedStation`):
* **`.onnx` + `.json`** — используются **наземной станцией `GeoWeedStation`** на ПК (CPU/GPU через ONNX Runtime). При необходимости их нужно перенести в `GeoWeedStation/models/`.
* **`.rknn`** (`*_rk3588_int8.rknn`) — квантованные модели для NPU бортового компьютера дрона (Orange Pi 5 Pro / Rockchip RK3588). Переносятся на **борт БПЛА** (в каталог бортового ПО, например `/home/orangepi/tsu/CollectOnboardImage/models/`), а не в наземную станцию.
* **`.pt`** — чекпоинты PyTorch для обучения, дообучения и оценки метрик в разделе `train/`.

#### Как `GeoWeedStation` подхватывает файлы в `GeoWeedStation/models/`

Внутри наземной станции модуль `OnnxDetector` автоматически сканирует `GeoWeedStation/models/`:

1. **Детекторы объектов (`.onnx` + `.json`)** — помещаются прямо в `GeoWeedStation/models/`:
   * В списке моделей отображаются **все `.onnx`-файлы** из `models/`, в имени которых нет маркеров классификаторов (`resnet`, `dualhead`, `_c07`, `_c06`).
   * Рядом с каждой моделью `<имя_модели>.onnx` должен лежать парный файл метаданных `<имя_модели>.json` (или `<имя_модели>_classes.json`; если их нет — используется `classes.json`). В JSON задаются классы, `model_type` (`yolo`/`yolox`), `input_profile` (`rgb`/`ms5`) и пороги.
   * Примеры файлов в `GeoWeedStation/models/`:
     * `d01_rgb_plant.onnx` + `d01_rgb_plant.json` — основной RGB-детектор растений (YOLOX-M);
     * `d04_ms5_plant.onnx` + `d04_ms5_plant.json` — 5-канальный мультиспектральный детектор для TIFF (`input_profile: "ms5"`);
     * `weed_detection_yolo.onnx` + `classes.json` — модель по умолчанию в [default_config.json](src/geo_weed_station/config/default_config.json);
     * `yolov8m.onnx` + `yolov8m.json` / `weed_multiclass_yolo.onnx` + `weed_multiclass_yolo.json` — дополнительные модели.

2. **Классификаторы второй стадии (`C07` и `C06`)** — помещаются в `GeoWeedStation/models/` под точными именами (или в подпапки):
   * Для **C07 (категории + фазы развития)**: файл **`GeoWeedStation/models/resnet50_dualhead_c07.onnx`** (также поддерживаются пути `models/c07_rgb_geosaero/c07_rgb_geosaero_best.onnx` и `models/best.onnx`).
   * Для **C06 (24 ботанических вида)**: файл **`GeoWeedStation/models/resnet50_c06.onnx`** (также поддерживается путь `models/c06_rgb_species/c06_rgb_species_best.onnx`).

Итоговая структура папки `GeoWeedStation/models/`:

```text
GeoWeedStation/models/
├── classes.json
├── d01_rgb_plant.onnx
├── d01_rgb_plant.json
├── d04_ms5_plant.onnx
├── d04_ms5_plant.json
├── resnet50_dualhead_c07.onnx
├── resnet50_c06.onnx
└── ... (все остальные файлы из папки готовых моделей)
```

> **Выбор модели в интерфейсе после запуска:**
> На правой панели в приложении выберите нужную модель детекции в блоке **«Нейросетевая модель»** (или укажите произвольный `.onnx` через кнопку `...`), а при необходимости включите вторую стадию в блоке **«Классификация видов»** (`Отключена` / `C07 (категории + фазы)` / `C06 (виды растений)`). Настройки также доступны через **Настройки → Нейросеть (ONNX)**.

---

### Шаг 3. Установка окружения

#### Вариант А: Автоматическая установка в один шаг (рекомендуется)

Репозиторий включает кроссплатформенные скрипты ([deploy/setup_linux.sh](deploy/setup_linux.sh), [deploy/setup_windows.bat](deploy/setup_windows.bat)), которые автоматически проверяют версию Python и системные библиотеки Qt/XCB (на Ubuntu/Debian), создают виртуальное окружение `.venv`, устанавливают зависимости, устраняют конфликты `opencv-python` с Qt (заменяя на `opencv-python-headless`) и генерируют готовые лаунчеры.

* **Linux (Ubuntu/Debian):**
  ```bash
  chmod +x setup.sh deploy/setup_linux.sh
  ./setup.sh
  # либо напрямую:
  ./deploy/setup_linux.sh
  ```

* **Windows 10/11 (CMD в корне репозитория):**
  ```cmd
  setup.bat
  :: либо напрямую:
  deploy\setup_windows.bat
  ```

#### Вариант Б: Ручная установка

```bash
# Создание и активация виртуального окружения
python3 -m venv .venv
source .venv/bin/activate        # на Windows: .venv\Scripts\activate

# Установка зависимостей и пакета в режиме разработки (editable)
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

---

### Шаг 4. Запуск приложения

#### Linux
```bash
# Стандартный запуск (с изоляцией Qt от конфликтов с темами GNOME/GTK/Wayland):
./run_geoweed.sh
# (также доступен алиас ./geoweedstation.sh)

# Запуск со встроенной самодиагностикой компонентов:
./run_geoweed.sh --selftest

# Headless-запуск (на сервере или в CI без дисплея):
./run_geoweed.sh --selftest --offscreen

# Запуск с пользовательским конфигом:
./run_geoweed.sh --config path/to/config.json
```

#### Windows
```cmd
:: Обычный запуск приложения (без консольного окна):
run_geoweed.bat

:: Отладочный запуск (с консольным окном и pause при завершении/сбое):
run_geoweed_debug.bat

:: Встроенная самодиагностика:
run_geoweed_debug.bat --selftest --offscreen
```

#### Прямой запуск через Python (из активированного `.venv`)
```bash
python main.py
# или:
python main.py --selftest --offscreen
```

---

## 🧪 Тестирование

Проект содержит модульные, интеграционные, нагрузочные (НФТ) и архитектурные тесты:

```bash
# Активация окружения (если запускаете pytest вручную)
source .venv/bin/activate  # на Windows: .venv\Scripts\activate

# Быстрый прогон модульных тестов
pytest -m "not slow"

# Полный прогон в headless-режиме (Linux)
QT_QPA_PLATFORM=offscreen pytest

# Запуск сквозных E2E-тестов с мок-сервером борта
pytest -m "e2e"

# Проверка нефункциональных требований (производительность, время отклика)
pytest -m "nfr"

# Проверка соблюдения архитектурных слоев (AST-анализ импортов)
pytest tests/architecture/test_import_rules.py

# Проверка интеграции 5-канального детектора D04 MS5 и классификации C06/C07
pytest tests/test_ms5_integration.py tests/test_onnx_classification.py
```

---

## 📦 Сборка и дистрибуция

Перед сборкой автономного бинарного пакета убедитесь, что файлы моделей (`.onnx` и `.json`) уже перенесены в `GeoWeedStation/models/` — скрипт сборки автоматически включает каталог `models/` внутрь дистрибутива ([deploy/build_app.py](deploy/build_app.py)).

```bash
# Сборка автономного приложения через PyInstaller
python deploy/build_app.py
```

Готовый исполняемый пакет формируется в каталоге `dist/GeoWeedStation/`.

---

## 📚 Документация, отчеты и обучение моделей

* [Сводный Roadmap разработки (Задачи 01–21)](docs/tasks/README.md)
* [Архитектура системы и слои](docs/architecture.md)
* [Руководство оператора станции](docs/user_guide.md)
* [План связи и запуска дрона Skyris (SSH/HTTP/SFTP)](docs/drone_communication_and_launch_plan.md)
* [Интеграция 5-канальной модели D04 MS5](docs/d04_ms5_plant_integration.md)
* [Спецификация метаданных XMP](docs/xmp_specification.md)
* [Аудит сторонних лицензий](docs/third_party_licenses.md)
* [Эмулятор бортового комплекса (Mock Server)](tests/mocks/README.md)
* [Автономный комплекс обучения моделей и метрики](train/standalone_training/README.md) · [Оценка на GeosAero](train/standalone_training/GEOSAERO_METRICS.md)
* [Отчет по Модулю A (PDF)](reports/Отчет%20модуль%20А.pdf) · [Отчет по Модулю B (DOCX)](reports/Modul_Б_.docx) · [Отчет по Модулю C (PDF)](reports/Отчет%20модуль%20C.pdf)
