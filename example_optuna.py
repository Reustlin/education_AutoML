import optuna
import numpy as np
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
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # создайте и обучите модель случайного леса с предложенными гиперпараметрами
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
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
study_rf.optimize(objective_rf, n_trials=10)

print('Лучшие параметры для RandomForest:', study_rf.best_params)

# определяем функцию objective для CatBoost

def objective_cb(trial):
    # предложите гиперпараметры модели CatBoost
    iterations = trial.suggest_int("iterations", 100, 1000)
    depth = trial.suggest_int("depth", 4, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)

    # создайте и обучите модель CatBoost с предложенными гиперпараметрами
    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
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
study_cb.optimize(objective_cb, n_trials=10)

print('Лучшие параметры для CatBoost:', study_cb.best_params)
