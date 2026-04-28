


# 📸 E-Commerce Visual Quality Analyzer & Price Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-F37626)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50-EE4C2C?logo=pytorch)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC?logo=tailwind-css)
![License](https://img.shields.io/badge/License-MIT-green)

> **Полноценный End-to-End Machine Learning проект**, который оценивает качество фотографий товаров и предсказывает их потенциальную кликабельность (CTR) и оптимальную рыночную цену.

---

## 📖 О проекте (Бизнес-ценность)

В сфере e-commerce качество визуального контента напрямую влияет на конверсию. Плохо освещенные, размытые или неудачно скомпонованные фотографии снижают доверие покупателей, уменьшают кликабельность (CTR) и заставляют продавцов демпинговать цены.

Этот проект решает проблему **автоматического аудита контента**. Система анализирует загруженную фотографию товара и с помощью алгоритмов компьютерного зрения (Computer Vision) и машинного обучения дает продавцу мгновенную обратную связь:
1. **Предсказывает ожидаемый CTR** (%).
2. **Оценивает оптимальную стоимость** товара на основе визуала ($).
3. Выявляет технические дефекты (размытие, пересвет, шум).

Проект построен как готовый к деплою SaaS-микросервис с REST API и современным веб-интерфейсом.

---

## ✨ Ключевые возможности

*   **Извлечение базовых CV-признаков:** Анализ яркости, контрастности и резкости (Variance of Laplacian) с помощью OpenCV.
*   **Глубокое обучение (Deep Learning):** Использование предобученной CNN `ResNet50` (PyTorch) для извлечения семантических эмбеддингов изображения.
*   **ML-Инференс:** Предсказание целевых метрик (Price, CTR) с помощью обученных моделей `XGBoost`.
*   **Быстрый Backend:** Асинхронный REST API на базе `FastAPI`.
*   **Современный UI:** Адаптивный веб-интерфейс с Drag-and-Drop загрузкой, написанный на Vanilla JS + Tailwind CSS (без тяжелых фреймворков).

---

## 🏗 Архитектура и Стек технологий

*   **Data Science & ML:** Python, Pandas, Scikit-learn, XGBoost.
*   **Computer Vision:** OpenCV, Pillow, PyTorch, Torchvision (ResNet50).
*   **Backend:** FastAPI, Uvicorn, Pydantic.
*   **Frontend:** HTML5, JavaScript, Tailwind CSS (via CDN).

---

## 📂 Структура репозитория

```text
photo_quality_ml/
│
├── data/                       # Директория для данных
│   ├── images/                 # Сгенерированные/реальные изображения товаров
│   └── dataset.csv             # Разметка датасета (image_path, CTR, Price)
│
├── models/                     # Обученные артефакты (.pkl / .joblib)
│   ├── ctr_model.joblib        # Модель XGBoost для предсказания CTR
│   ├── price_model.joblib      # Модель XGBoost для предсказания Цены
│   └── scaler.joblib           # StandardScaler для нормализации фичей
│
├── src/                        # Исходный код ML-пайплайна
│   ├── generate_data.py        # Скрипт генерации синтетических данных
│   ├── feature_extractor.py    # Логика извлечения CV-фичей и эмбеддингов ResNet
│   └── train.py                # Пайплайн обучения и валидации моделей
│
├── app.py                      # FastAPI приложение (REST API)
├── index.html                  # Frontend (UI для пользователя)
├── requirements.txt            # Зависимости Python
└── README.md                   # Документация проекта
```

---

## 🚀 Быстрый старт (Локальный запуск)

Следуйте этим шагам, чтобы запустить проект на своем компьютере.

### 1. Клонирование репозитория и настройка окружения

```bash
# Клонируем репозиторий
git clone https://github.com/ВАШ_USERNAME/photo_quality_ml.git
cd photo_quality_ml

# Создаем виртуальное окружение (рекомендуется)
python -m venv venv

# Активируем окружение (Windows)
venv\Scripts\activate
# Активируем окружение (macOS / Linux)
source venv/bin/activate

# Устанавливаем зависимости
pip install -r requirements.txt
```

### 2. Подготовка данных и обучение ML-модели

Поскольку репозиторий не содержит тяжелых весов моделей, вам нужно обучить их локально. Пайплайн полностью автоматизирован:

```bash
# Шаг 2.1: Генерируем синтетический датасет (500 изображений + CSV)
python src/generate_data.py

# Шаг 2.2: Запускаем извлечение фичей и обучение XGBoost моделей
python src/train.py
```
*После выполнения `train.py` в папке `models/` появятся готовые к работе файлы `.joblib`.*

### 3. Запуск веб-приложения

```bash
# Запускаем FastAPI сервер с помощью Uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

*   **API Документация (Swagger):** Доступна по адресу `http://localhost:8000/docs`
*   **Пользовательский интерфейс (UI):** Откройте файл `index.html` прямо в браузере (двойной клик) или раздайте его через любой локальный сервер (например, Live Server в VS Code). Введите в UI адрес бэкенда: `http://localhost:8000`.

---

## 🧠 Под капотом: Как работает Feature Engineering?

Чтобы модель XGBoost могла понимать изображения, сырые пиксели конвертируются в табличные данные (векторы) на этапе извлечения признаков (`src/feature_extractor.py`):

1.  **Blur Score (Оценка размытия):** Считается через дисперсию оператора Лапласа `cv2.Laplacian(img, cv2.CV_64F).var()`. Низкие значения указывают на не в фокусе (размытые) фото.
2.  **Brightness / Contrast:** Расчет среднего значения пикселей в градациях серого и их стандартного отклонения.
3.  **Deep Semantic Features (Эмбеддинги):** Фотография прогоняется через предобученную на ImageNet модель `ResNet50`. Мы отбрасываем последний слой классификации и забираем вектор из 2048 чисел. Для ускорения обучения и снижения размерности (Dimensionality Reduction) мы можем применять PCA.

Эти фичи конкатенируются и подаются на вход `XGBRegressor`, который предсказывает непрерывные значения CTR и Price.

---

## 🔮 Roadmap / Планы по развитию

- [ ] Заменить синтетические данные на реальный датасет маркетплейса (например, с Kaggle: Shopee Product Data).
- [ ] Добавить удаление фона (Background Removal) с помощью библиотеки `rembg` для расчета полезной площади, занимаемой товаром на фото.
- [ ] Обернуть проект в Docker Compose (создать `Dockerfile` для FastAPI и `nginx` для отдачи статики).
- [ ] Настроить CI/CD пайплайн через GitHub Actions для деплоя на AWS EC2 / Render.

---

## 👨‍💻 Автор

**Ваше Имя / Никнейм** 
* Machine Learning Engineer / Data Scientist
* GitHub: [@ВашUsername](https://github.com/ВашUsername)
* LinkedIn: [ВашПрофиль](https://linkedin.com/in/ВашПрофиль)

Если вам понравился этот проект, пожалуйста, поставьте ⭐️ репозиторию!

---

---

