import os
import pandas as pd
import re
import rtf_to_text
 
def extract_data_from_file(file_path):
    # Open the .doc file using the with statement
    with open(file_path, 'rb') as file:
        content = file.read()
        text = rtf_to_text(content)
 
    # Extract FIO (Full Name)
    fio_match = re.search(r'\|(.+?) \n', text)
    fio = fio_match.group(1) if fio_match else None
 
    # Extract City
    city_pattern = r'Проживает: (.*?)\n'
    city_match = re.search(city_pattern, text)
    city = city_match.group(1) if city_match else ''
 
    # Extract Work Experience
    experience_pattern = r'Опыт работы —(.*?)Образование'
    experience_match = re.search(experience_pattern, text, re.DOTALL)
    experience = experience_match.group(1) if experience_match else ''
 
    # Extract Education
    education_pattern = r'Образование(.*?)((Повышение квалификации, курсы)|(Тесты, экзамены)|(Ключевые навыки))'
    education_match = re.search(education_pattern, text, re.DOTALL)
    education = education_match.group(1) if education_match else ''
 
    # Extract Qualification and Courses
    qualification_pattern = r'Повышение квалификации, курсы(.*?)((Тесты, экзамены)|(Ключевые навыки)|(Опыт вождения))'
    qualification_match = re.search(qualification_pattern, text, re.DOTALL)
    qualification = qualification_match.group(1) if qualification_match else ''
 
    # Create a dictionary to store the extracted data
    data = {'ФИО': [fio], 'Город': [city], 'Опыт работы': [experience.strip()], 'Образование': [education.strip()], 'Повышение квалификации, курсы': [qualification.strip()]}
 
    return data
 
def process_folder(folder_path):
    # Create an empty DataFrame
    df = pd.DataFrame()
 
    # Iterate over all files in the folder
    for file in os.listdir(folder_path):
        if file.endswith('.doc'):
            file_path = os.path.join(folder_path, file)
            data = extract_data_from_file(file_path)
            # Append the extracted data to the DataFrame
            df = df.append(pd.DataFrame(data))
 
    return df
 
# Specify the folder path
folder_path = '/path/to/folder'
 
# Process the folder and get the resulting DataFrame
df = process_folder(folder_path)
 
print()

import re

def filter_words(text):
    pattern = r'\ b[A-ZА-ЯЁ][a-zа-яё]*\b'
    return re.findall(pattern, text)

text = "Today is Thursday, August 29, 2024 and here are the results:"
print(filter_words(import streamlit as st
import os

# Создайте папку для сохранения файлов
save_folder = 'uploads'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Загрузите файл
uploaded_file = st.file_uploader('Выберите файл')

# Если файл загружен, сохраните его в папку
if uploaded_file is not None:
    file_path = os.path.join(save_folder, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())
    st.success('Файл сохранен!')

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

# Пример с данными iris
X, y = load_iris(return_X_y=True)

# Определяем модель
model = SVC()

# Определяем гиперпараметры для подбора
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Определяем кросс-валидацию
cv = StratifiedKFold(n_splits=5)

# Создаем объект GridSearchCV с return_train_score=True
grid_search = GridSearchCV(model, param_grid, cv=cv, return_train_score=True)

# Обучаем модель
grid_search.fit(X, y)

# Извлекаем результаты кросс-валидации
results = grid_search.cv_results_

# Выводим значения на каждом фолде с тестовой выборки
for fold_idx in range(cv.get_n_splits()):
    test_scores = results['split{}_test_score'.format(fold_idx)]
    print(f"Фолд {fold_idx + 1}: {test_scores}")


Да, совершенно верно! 

В коде, который я вам предоставил, 'x' и 'y' - это просто примеры имен столбцов, которые используются для поиска ближайших соседей. Вам нужно заменить эти имена на фактические имена столбцов в ваших датафреймах, которые содержат признаки, по которым нужно искать ближайших соседей. 

Вот как это сделать:

1. Замените 'x' и 'y': 
    * В строке knn.fit(df2[['x', 'y']]) замените 'x' и 'y' на имена столбцов в df2, которые содержат ваши признаки.
    * В строке distances, indices = knn.kneighbors(df1[['x', 'y']]) замените 'x' и 'y' на имена столбцов в df1, которые содержат ваши признаки.

2. Проверьте типы данных: 
    * Убедитесь, что столбцы, которые вы используете для поиска ближайших соседей, имеют подходящий тип данных для выбранной метрики расстояния. Например, если вы используете евклидово расстояние (metric='euclidean'), то столбцы должны быть числовыми.

Вот пример того, как это может выглядеть с другими именами столбцов:

import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Создаем два датафрейма
df1 = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
df2 = pd.DataFrame({'feature1': [1.5, 2.5, 3.5], 'feature2': [4.5, 5.5, 6.5]})

# Создаем объект NearestNeighbors
knn = NearestNeighbors(n_neighbors=1, metric='euclidean') 

# Обучаем модель на датафрейме 2
knn.fit(df2[['feature1', 'feature2']])

# Находим ближайших соседей для каждой строки в df1
distances, indices = knn.kneighbors(df1[['feature1', 'feature2']])

# Создаем новый датафрейм с результатами
results = pd.DataFrame({'df1_index': df1.index, 'df2_index': indices.flatten()})

# Добавляем столбцы с информацией о ближайших соседях
results = results.merge(df1, left_on='df1_index', right_index=True)
results = results.merge(df2, left_on='df2_index', right_index=True, suffixes=('_df1', '_df2'))

# Выводим результаты
print(results)



2,622/12,000


# Открываем файл для добавления форматирования
wb = openpyxl.load_workbook(excel_path)
ws = wb.active

# Применяем форматирование к каждому столбцу
for col in ws.columns:
    max_length = 0
    column = col[0].column  # Номер столбца

    for cell in col:
        # Устанавливаем перенос текста и выравнивание
        cell.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
        # Измеряем длину содержимого для установки ширины
        max_length = max(max_length, len(str(cell.value) or ""))

    # Устанавливаем ширину столбца на основе максимальной длины текста
    adjusted_width = max_length + 2
    ws.column_dimensions[get_column_letter(column)].width = adjusted_width

# Сохраняем изменения
wb.save(excel_path)
wb.close()
