# GeoWeedStation — Roadmap разработки

Порядок слоёв: инфраструктура → схемы → Core → сеть и мок → воркеры → UI → контроллеры/интеграция → deploy и E2E.
Источники: [technical specification.md](../technical%20specification.md), [agrovision_onboard_rest_api_specification.md](../agrovision_onboard_rest_api_specification.md).

| № | Название | Зависимости | Ключевые файлы | Статус |
|---|----------|-------------|----------------|--------|
| [01](task_01_infrastructure_config.md) | Инфраструктура и конфигурация | — | `pyproject.toml`, `requirements.txt`, `config/settings.py`, `main.py` | Completed |
| [02](task_02_schemas_dto.md) | Pydantic-схемы и DTO | 01 | `schemas/entities.py`, `payload.py`, `mission.py`, `telemetry.py`, `onboard_api.py` | Completed |
| [03](task_03_xmp_parser.md) | Парсер XMP | 01, 02 | `core/xmp_parser.py` | Completed |
| [04](task_04_onnx_detector.md) | ONNX-детектор | 01, 02 | `core/detector_base.py`, `core/onnx_detector.py` | Completed |
| [05](task_05_map_engine_projection.md) | MapEngine: проекция и СК | 01, 02 | `map_engine/projector.py`, `coordinates.py` | Completed |
| [06](task_06_map_engine_heatmap_export.md) | MapEngine: heatmap, статистика, экспорт | 02, 05 | `map_engine/weed_map.py`, `heatmap.py`, `exporter.py` | Completed |
| [07](task_07_mission_planner.md) | Планировщик миссий | 02 | `core/mission_planner.py` | Completed |
| [08](task_08_network_client.md) | REST-клиент и менеджер загрузок | 01, 02 | `network/drone_client.py`, `download_manager.py`, `exceptions.py` | Completed |
| [09](task_09_mock_onboard_server.md) | Мок-сервер борта | 02, 08 | `tests/mocks/mock_onboard_server.py` | Completed |
| [10](task_10_workers_online.md) | Воркеры онлайн (QThread) | 01, 02, 08, 09 | `workers/telemetry_worker.py`, `command_worker.py`, `download_worker.py` | Completed |
| [11](task_11_workers_offline.md) | Воркеры офлайн | 02–06, 10 | `workers/batch_image_worker.py`, `video_worker.py`, `core/source_factory.py` | Completed |
| [12](task_12_ui_shell.md) | UI-каркас | 01, 02, 10 | `ui/main_window.py`, `widgets/status_bar_panel.py`, `styles.qss` | Completed |
| [13](task_13_ui_viewers.md) | UI: просмотр и видеоплеер | 02, 11, 12 | `widgets/image_viewer.py`, `video_player_widget.py`, `file_list_panel.py` | Completed |
| [14](task_14_ui_map_mission.md) | UI: карта и управление миссией | 02, 05–07, 12 | `widgets/map_canvas.py`, `mission_control.py`, `map_info_panel.py`, `telemetry_panel.py`, `runs_table.py` | Completed |
| [15](task_15_ui_dialogs.md) | UI: диалоги | 01, 06, 12 | `dialogs/settings_dialog.py`, `export_dialog.py`, `confirm_dialog.py` | Completed |
| [16](task_16_controllers_offline.md) | Контроллеры: офлайн, карта, экспорт | 03, 04, 06, 11–15 | `controllers/offline_controller.py`, `map_controller.py`, `export_controller.py`, `app_controller.py` | Completed |
| [17](task_17_controllers_online.md) | Контроллеры: онлайн и менеджер запусков | 07–11, 14–16 | `controllers/online_controller.py`, `runs_controller.py` | Completed |
| [18](task_18_packaging_docs.md) | Упаковка и документация | 12, 16, 17 | `deploy/build_app.py`, `installer_linux.sh`, `docs/*.md` | Completed |
| [19](task_19_e2e_integration_tests.md) | Сквозные тесты и НФТ | 16, 17, 18 | `tests/e2e/`, `tests/nfr/`, `tests/architecture/` | Completed |
| [20](task_20_weed_classification_stage.md) | Вторая стадия классификации (C06/C07) | 04, 16 | `core/onnx_detector.py`, `ui/widgets/detection_filter_panel.py` | Completed |
| [21](../d04_ms5_plant_integration.md) | Интеграция D04 MS5 (5-канальный TIFF) | 04, 11, 16, 20 | `core/image_io.py`, `core/onnx_detector.py`, `workers/ms5_preview_worker.py` | Completed |

## Принятые решения и расхождения документов
- При конфликте ТЗ и спецификации REST приоритет у спецификации борта: `use_nn` (не `use_model`), `save_policy`, `action=rth` (не `return`), `type=summary|archive|image` + `wp_index`, `coordinate_system="aruco_local"`.
- Внутри станции (XMP, MapEngine) используется `local_aruco`; маппинг имён СК — только в `schemas/onboard_api.py`.
- Добавлен пакет `controllers/` (MVC), отсутствующий в дереве ТЗ.
- Частота опроса: ТЗ даёт 1–2 Гц (0.5–1.0 с), таймаут ≤ 2 с.
