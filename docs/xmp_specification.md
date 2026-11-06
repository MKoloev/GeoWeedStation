# Спецификация метаданных XMP в GeoWeedStation

## 1. Общие принципы работы с метаданными

В соответствии с техническим заданием GeoWeedStation извлечение пространственных параметров съемки (высота, ориентация подвеса, фокусное расстояние) выполняется **исключительно из встроенных XMP-пакетов** внутри графических файлов (JPEG / TIFF).

> [!IMPORTANT]
> Использование любых внешних sidecar-файлов (`.txt`, `.json`, `.xml` рядом с файлами изображений) **строго запрещено**.

---

## 2. Поддерживаемые пространства имен XMP

| Префикс | URI пространства имен | Назначение |
|---|---|---|
| `drone-dji` | `http://www.dji.com/drone-dji/1.0/` | Высота, углы ориентации дрона и подвеса |
| `Camera` | `http://pix4d.com/camera/1.0/` | Оптические параметры матрицы и FOV |
| `tiff` / `exif` | `http://ns.adobe.com/tiff/1.0/` | Размеры матрицы, фокусное расстояние |
| `agro` | `http://agrovision.aero/xmp/1.0/` | Локальные координаты полигона ArUco |

---

## 3. Таблица ключевых тегов геопривязки

| XMP-тег | Тип данных | Единицы | Описание |
|---|---|---|---|
| `drone-dji:RelativeAltitude` | `float` | метры | Относительная высота $H$ над точкой взлета |
| `drone-dji:AbsoluteAltitude` | `float` | метры | Абсолютная высота над уровнем моря (AMSL) |
| `drone-dji:GimbalYawDegree` | `float` | градусы | Курс оптической оси подвеса камеры (относительно Севера) |
| `drone-dji:GimbalPitchDegree`| `float` | градусы | Угол тангажа подвеса (-90° — надир) |
| `drone-dji:GimbalRollDegree` | `float` | градусы | Угол крена подвеса |
| `drone-dji:FlightYawDegree`  | `float` | градусы | Курс фюзеляжа БПЛА (используется при отсутствии подвеса) |
| `exif:FocalLength`          | `float` | мм | Фокусное расстояние объектива |
| `Camera:SensorWidth`        | `float` | мм | Физическая ширина светочувствительной матрицы |
| `Camera:SensorHeight`       | `float` | мм | Физическая высота светочувствительной матрицы |
| `agro:LocalCoordinateX`     | `float` | метры | Локальная координата X в метрах полигона ArUco |
| `agro:LocalCoordinateY`     | `float` | метры | Локальная координата Y в метрах полигона ArUco |

---

## 4. Пример структуры встроенного XML/XMP пакета

```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
        xmlns:Camera="http://pix4d.com/camera/1.0/"
        xmlns:agro="http://agrovision.aero/xmp/1.0/"
        drone-dji:RelativeAltitude="+15.40"
        drone-dji:AbsoluteAltitude="+142.80"
        drone-dji:GimbalYawDegree="+35.2"
        drone-dji:GimbalPitchDegree="-90.0"
        drone-dji:GimbalRollDegree="0.0"
        drone-dji:FlightYawDegree="+35.0"
        Camera:SensorWidth="6.4"
        Camera:SensorHeight="4.8"
        agro:LocalCoordinateX="14.25"
        agro:LocalCoordinateY="28.90">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

## 5. Поведение при отсутствии или невалидности тегов

Если в изображении отсутствуют теги `RelativeAltitude` (или $H \le 0$), либо отсутствуют углы ориентации или плановые координаты:
1. Кадр помечается статусом `NO_METADATA` в пакетном списке.
2. Модуль `MapEngine` блокируется (`is_locked = True`).
3. Вкладка цифровой карты блокируется с сообщением оператору:
   *«Геопривязка отсутствует. Доступен только просмотр исходных изображений»*.
4. Разрешается только просмотр исходного изображения и наложение рамок детекции в пикселях.
