import optuna
import numpy as np
import pickle
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# загружаем датасет
data = load_diabetes()
X, y = data.data, data.target

# разбиваем на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# определяем функцию objective для RandomForest
def objective_rf(trial):
    # предложите гиперпараметры модели случайного леса
    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth = trial.suggest_int("max_depth", 2, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # создайте и обучите модель случайного леса с предложенными гиперпараметрами
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=42
    )
    model.fit(X_train, y_train)

    # предсказание и вычисление метрики MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return -mse  # верните значение отрицательного MSE для минимизации

# создайте и оптимизируйте study для RandomForest
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=50)

print('Лучшие параметры для RandomForest:', study_rf.best_params)

# обучаем модель RandomForest с лучшими параметрами
best_rf_model = RandomForestRegressor(**study_rf.best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

# сохраняем модель RandomForest в pkl-файл
with open("best_rf_model.pkl", "wb") as f:
    pickle.dump(best_rf_model, f)

# определяем функцию objective для CatBoost
def objective_cb(trial):
    # предложите гиперпараметры модели CatBoost
    iterations = trial.suggest_int("iterations", 100, 2000)
    depth = trial.suggest_int("depth", 4, 16)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 20)
    border_count = trial.suggest_int("border_count", 32, 255)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 10)
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])

    # создайте и обучите модель CatBoost с предложенными гиперпараметрами
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        grow_policy=grow_policy,
        random_state=42,
        verbose=0
    )
    model.fit(X_train, y_train)

    # предсказание и вычисление метрики MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return -mse  # верните значение отрицательного MSE для минимизации

# создайте и оптимизируйте study для CatBoost
study_cb = optuna.create_study(direction="maximize")
study_cb.optimize(objective_cb, n_trials=50)

print('Лучшие параметры для CatBoost:', study_cb.best_params)

# обучаем модель CatBoost с лучшими параметрами
best_cb_model = CatBoostRegressor(**study_cb.best_params, random_state=42, verbose=0)
best_cb_model.fit(X_train, y_train)

# сохраняем модель CatBoost в pkl-файл
with open("best_cb_model.pkl", "wb") as f:
    pickle.dump(best_cb_model, f)

print("Модели сохранены в файлы 'best_rf_model.pkl' и 'best_cb_model.pkl'.")
