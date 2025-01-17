# Проект по распознаванию кожного заболевания на изображении

### Структура проекта

```bash

.
├── README.md
├── conf
│   ├── config.yaml
│   ├── dataset
│   │   └── dataset.yaml
│   ├── mlflow
│   │   └── mlflow.yaml
│   └── training
│       └── training.yaml
├── data
│   ├── test.dvc
│   └── train.dvc
├── poetry.lock
├── pyproject.toml
└── skin-diseases
    ├── __init__.py
    ├── dataset.py
    ├── infer.py
    ├── metrics.py
    └── train.py


```

### Формулировка задачи

Задача заключается в создании многоклассового (23 класса) классификатора для предсказания кожных 
заболеваний по изображению. Применение машинного обучения и нейросетей очень актуально в медицине и может сильно врачам в определении диагнозов больных.

### Данные

Изображения взяты с публичного портала [Dermnet](https://dermnet.com/) ([Ссылка на данные с Kaggle](https://www.kaggle.com/datasets/shubhamgoel27/dermnet)), который является крупнейшим источником информации по дерматологии, созданным с целью предоставления медицинского онлайн-образования.

Данные включают в себя изображения 23-х типов кожных заболеваний, таких как акне, экзема, меланома, псориаз и т.д. 
Общее количество изображений составляет около 19500, из которых примерно 15500 в обучающей выбокре, а остальные - в тестовой. Данных достаточно много, их должно хватить 
для решения задачи. 

Главной проблемой представленных данных является сильный дисбаланс классов, где изображений самого популярного класса в 7 раз больше самого непопулярного. Возможное решение проблемы - ансемблинг и аугментации.

### Подход к моделированию

Для решенеия задачи я планирую попробовать следующие архитектуры нейронных сетей:

- EfficientNetV2-S
- RegNetY-8GF
- ConvNeXt Tiny
- DenseNet-161
- ResNeXt-50 32x4d

Они показывают довольно хорошее качество при небольших требованиях к вычислительным ресурсам. Предварительно к обучающей выборке применим аугментации для избавления от дисбаланса классов и расширения выборки.

### Способ предсказания

Продакшен пайплайн будет представлять из себя вебсайт, написанный с помощью библиотек FastAPI и Streamlit. Модель будет делать предсказания, предварительно применив 
предобработку к изображению (ресайз, приведение к чёрно-белому цвету).


