import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

# Настройки путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH = os.path.join(DATA_DIR, 'dataset.csv')

# Создаем папки, если их нет
os.makedirs(IMAGES_DIR, exist_ok=True)

def create_synthetic_image(filepath, quality='good'):
    """Генерирует картинку с 'товаром' и применяет искажения в зависимости от качества"""
    # Создаем базовый фон (светло-серый)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 220
    
    # Рисуем "товар" (случайный цветной прямоугольник или круг по центру)
    color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    if random.choice([True, False]):
        cv2.rectangle(img, (75, 75), (225, 225), color, -1)
    else:
        cv2.circle(img, (150, 150), 75, color, -1)

    # Добавляем детали (имитация текстуры)
    cv2.putText(img, "PRODUCT", (90, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Искажения в зависимости от флага качества
    if quality == 'blur':
        # Размытие (убивает sharpness)
        img = cv2.GaussianBlur(img, (15, 15), 0)
    elif quality == 'dark':
        # Сильное затемнение (убивает brightness/exposure)
        img = (img * 0.4).astype(np.uint8)
    elif quality == 'noise':
        # Добавление шума (плохая камера)
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    # Сохраняем (в BGR формате, как работает OpenCV)
    cv2.imwrite(filepath, img)

def generate_dataset(num_samples=500):
    """Генерирует датасет изображений и CSV разметку"""
    print(f"Генерация {num_samples} изображений...")
    data = []
    
    qualities = ['good', 'blur', 'dark', 'noise']
    
    for i in tqdm(range(num_samples)):
        filename = f"item_{i}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        # Выбираем случайное качество (70% хороших, 30% плохих)
        quality = random.choices(qualities, weights=[0.7, 0.1, 0.1, 0.1])[0]
        
        create_synthetic_image(filepath, quality)
        
        # Генерируем таргеты (CTR и Цена) на основе качества фото
        # Базовая цена $10-$200, базовый CTR 1% - 15%
        if quality == 'good':
            ctr = round(random.uniform(5.0, 15.0), 2)  # Высокий CTR
            price = round(random.uniform(50.0, 200.0), 2) # Высокая цена
        else:
            ctr = round(random.uniform(0.5, 4.0), 2)   # Низкий CTR из-за плохого фото
            price = round(random.uniform(10.0, 60.0), 2)  # Низкая цена
            
        data.append({
            'image_path': f"data/images/{filename}",
            'quality_label': quality, # Для дебага, в обучении использовать не будем
            'ctr_percent': ctr,
            'price_usd': price
        })
        
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH, index=False)
    print(f"Датасет успешно сохранен в {CSV_PATH}")

if __name__ == "__main__":
    generate_dataset(num_samples=500) # 500 картинок хватит для пруф-оф-концепта