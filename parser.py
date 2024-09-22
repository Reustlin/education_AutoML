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


