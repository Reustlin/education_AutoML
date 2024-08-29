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
 
print(df)