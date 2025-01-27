import cv2
import numpy as np

# Функция для обработки одного изображения
def process_image(image_path, roi_coords, threshold, white_pixel_limit):
    # Шаг 1. Загрузка изображения
    image = cv2.imread(image_path)
    
    # Шаг 2. Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Шаг 3. Сглаживание
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Шаг 4. Выделение ROI
    x, y, w, h = roi_coords
    roi = blurred[y:y+h, x:x+w]
    
    # Шаг 5. Пороговое преобразование
    _, binary = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
    
    # Шаг 6. Подсчёт белых пикселей
    white_pixels = cv2.countNonZero(binary)
    
    # Шаг 7. Логика классификации
    if white_pixels > white_pixel_limit:
        print(f"Перелив обнаружен! Белых пикселей: {white_pixels}")
    else:
        print(f"Перелива нет. Белых пикселей: {white_pixels}")
    
    # Шаг 8. Визуализация
    cv2.imshow("Original Image", image)
    cv2.imshow("ROI", roi)
    cv2.imshow("Binary ROI", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Координаты ROI (пример: x, y, ширина, высота)
roi_coords = (100, 200, 300, 100)  # Настрой для каждой камеры

# Пороговое значение и лимит белых пикселей
threshold = 200  # Подобрать вручную
white_pixel_limit = 500  # Настроить по тестовым данным

# Тестовое изображение
image_path = "example.jpg"  # Укажи путь к тестовому изображению

# Обработка изображения
process_image(image_path, roi_coords, threshold, white_pixel_limit)
