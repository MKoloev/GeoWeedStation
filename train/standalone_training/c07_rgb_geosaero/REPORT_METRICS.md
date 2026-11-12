C07 оценена на 274 объектах датасета GeosAero. Для категорий accuracy составила 70,4380%, balanced accuracy 56,7697%, macro-F1 61,8352%. Для стадий роста accuracy составила 59,8540%, balanced accuracy 87,1495%, macro-F1 73,6740%. Accuracy показывает долю правильных ответов, balanced accuracy учитывает классы с одинаковым весом, macro-F1 отражает баланс ложных отнесений и пропусков по классам. Категория и стадия одновременно определены верно у 56,2044% объектов. Оценка выполнена на эталонных кропах из датасета обучения.

Пути относительно корня проекта:
Веса: runs/c07_rgb_geosaero/best.pt
Инференс: standalone_training/c07_rgb_geosaero/infer.py
Результаты: standalone_training/c07_rgb_geosaero/geosaero_evaluation/metrics.json
