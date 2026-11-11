# TASK 20 — Этап нейросетевой классификации сорных растений (ResNet-50 C07 Dual-Head)

## ID и название
`T-20` · Двухстадийный инференс сорняков: интеграция ONNX-классификатора видов и фенологических стадий (ResNet-50 C07) в `OnnxDetector`.

## Зависимости (Prerequisites)
T-02, T-04, T-11, T-13, T-16.

## Целевые файлы
- `src/geo_weed_station/schemas/entities.py` (расширение `DetectedWeed` полями `stage_name`, `class_prob`)
- `src/geo_weed_station/config/settings.py` (параметры классификатора в `DetectorSettings`)
- `src/geo_weed_station/core/onnx_detector.py` (вырезка кропов с 10% padding, нормализация ImageNet, вызов `resnet50_dualhead_c07.onnx`)
- `src/geo_weed_station/core/annotation_io.py` (сериализация/десериализация стадий)
- `src/geo_weed_station/ui/widgets/image_viewer.py` и `video_player_widget.py` (отображение видов и стадий)
- `tests/test_onnx_classification.py`
- `docs/task_classification_agent_prompt.md`

## Архитектурные требования и контракты
Полная спецификация и пошаговые инструкции описаны в [docs/task_classification_agent_prompt.md](../task_classification_agent_prompt.md).

1. **Входная предобработка кропа:**
   - Рамка детекции расширяется на $10\%$ с ограничением границами изображения.
   - Resize до $224 \times 224$, RGB, нормализация ImageNet ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
   - Формат тензора NCHW: `(1, 3, 224, 224)` float32.
2. **Инференс ONNX Runtime:**
   - Модель `resnet50_dualhead_c07.onnx`.
   - Вход `images`, выходы `logits_category` $[1, 11]$ и `logits_stage` $[1, 5]$.
   - Softmax декодирование с вычислением `cat_id`, `stage_id` и вероятностей.
3. **Виртуальное окружение:**
   - Все проверки и запуски осуществляются в `.venv_gws`.

## Критерии приемки
- [ ] Все тесты `pytest tests/` в `.venv_gws` проходят без регрессий.
- [ ] Опция классификатора переключается через `DetectorSettings` (`c07` / `none`).
- [ ] Отображение названий сорняков и стадий на русском языке в UI.
