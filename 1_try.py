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

import cv2
import numpy as np

def process_video(video_path, points, threshold, white_pixel_limit):
    # Шаг 1. Открытие видео
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Частота кадров
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Всего кадров
    duration = frame_count / fps  # Длительность видео
    
    print(f"Частота кадров: {fps} FPS")
    print(f"Всего кадров: {frame_count}")
    print(f"Длительность видео: {duration:.2f} секунд")
    
    # Инициализация счётчиков
    foam_frames = 0
    no_foam_frames = 0

    # Создание маски на основе ROI
    mask = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Шаг 2. Преобразование кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Шаг 3. Подсчёт белых пикселей
        white_pixels = cv2.countNonZero(binary)

        # Шаг 4. Классификация состояния
        if white_pixels > white_pixel_limit:
            foam_frames += 1
            status = "Есть пена"
            color = (0, 0, 255)  # Красный
        else:
            no_foam_frames += 1
            status = "Нет пены"
            color = (0, 255, 0)  # Зелёный

        # Шаг 5. Визуализация
        cv2.putText(frame, f"Состояние: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Белые пиксели: {white_pixels}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Video", frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Шаг 6. Подсчёт времени
    foam_time = foam_frames / fps
    no_foam_time = no_foam_frames / fps

    print(f"\nРезультаты анализа:")
    print(f"Время с пеной: {foam_time:.2f} секунд")
    print(f"Время без пены: {no_foam_time:.2f} секунд")

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

# Пример: координаты 4 точек ROI (по часовой стрелке)
points = [(100, 200), (400, 200), (350, 300), (150, 300)]  # Настроить вручную

# Пороговое значение и лимит белых пикселей
threshold = 200  # Подобрать вручную
white_pixel_limit = 500  # Настроить по тестовым данным

# Тестовое видео
video_path = "example_video.mp4"  # Укажите путь к видео

# Запуск обработки видео
process_video(video_path, points, threshold, white_pixel_limit)




