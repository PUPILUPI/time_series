import csv
import numpy as np
import pandas as pd
import seaborn as sb
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
import pmdarima as pm


# функция импорта даты
def import_data(path: str, sep: str):
    with open(path) as csv_file:
        read_data = pd.read_csv(path, sep=sep)
    return read_data


# функция конвертации значения ячейки в дату
def convert_data(data: pd.DataFrame):
    data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')
    return data['DATE']


# приведение значений временных рядов
def convert_value(data: pd.DataFrame):
    data['OPEN'] = data['OPEN'].apply(lambda x: x / data['OPEN'].mean())
    return data


# ф. вычисление стандартных характеристик рядов
def get_hist_stat(data: [pd.DataFrame], label_list: list, col_name: str = "OPEN"):
    fig, axes = plt.subplots(1, len(data), figsize=(16, 6))

    for i in range(len(data)):
        ax1 = data[i][col_name].hist(bins=100, ax=axes[i], color="lightblue")

        ax1.axvline(data[i][col_name].max(), color="black", label="max= " + str(data[i][col_name].max()))
        ax1.axvline(data[i][col_name].min(), color="black", label="min= " + str(data[i][col_name].min()))
        ax1.axvline(data[i][col_name].mean(), color="red", label="mean= " + str(data[i][col_name].mean()))

        std1 = np.std(data[i][col_name])
        ax1.axvline(data[i][col_name].mean() + std1, color="green", linestyle='--', label="+ 1 std= " + str(std1))
        ax1.axvline(data[i][col_name].mean() - std1, color="green", linestyle='--', label="- 1 std")

        ax1.legend(loc="upper right")
        label_list[
            i] += f"\nskew = {round(stats.skew(data[i][col_name]), 2)}. kurtosis = {round(stats.kurtosis(data[i][col_name]), 2)}"
        ax1.set_title(label_list[i])

    plt.legend()
    plt.show()


# ф. построения трендов
def show_trends(data: pd.DataFrame, label: str, degree: int):
    x = list(range(1, len(data) + 1))
    plt.plot(x, data['OPEN'], label=label)
    # линейный
    plt.plot(x, np.poly1d(np.polyfit(x, data['OPEN'], 1))(x), label='Линейный тренд')
    # Полиноминальный
    plt.plot(x, np.poly1d(np.polyfit(x, data['OPEN'], degree))(x), label='Полиноминальный тренд')
    # Экспоненциальный тренд
    plt.plot(x, np.exp(np.poly1d(np.polyfit(x, np.log(data['OPEN']), degree))(x)), label='Экспоненциальный тренд')
    plt.legend()
    plt.show()


# функция которая находит ряд приращений
def calc_returns(data: pd.DataFrame):
    returns = (data / data.shift(1)) - 1
    returns = returns.dropna()
    return returns


# подготовка данных для фосагро
phosAgro_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Ростовский\\PhosArgo.csv", ";");
convert_data(phosAgro_data)
convert_value(phosAgro_data)
phosAgro_data = phosAgro_data.set_index('DATE')
# для русагро
rusAgro_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Ростовский\\RusAgro.csv", ";");
convert_data(rusAgro_data)
convert_value(rusAgro_data)
rusAgro_data = rusAgro_data.set_index('DATE')
# для пшеницы
wheat_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Ростовский\\wheat.csv", ";");
convert_data(wheat_data)
convert_value(wheat_data)
wheat_data = wheat_data.set_index('DATE')

# Строим графики
plt.plot(phosAgro_data, label='ФосАгро')
plt.plot(rusAgro_data, label='РусАгро')
plt.plot(wheat_data, label='Пшеница')
plt.legend()
# вычисляем стандартные характеристики для рядов:
# Максимум и минимум,среднее, дисперсию, коэффициент ассиметрии, коэффициент эксцесса
# и строим их.
get_hist_stat([phosAgro_data, rusAgro_data, wheat_data], ["ФосАгро", "Русагро", "Пшеница"])
# накладываем тренды
fig2 = plt.figure()
# Показываем тренды
show_trends(phosAgro_data, "ФосАгро", 3)
show_trends(rusAgro_data, "РусАгро", 6)
show_trends(wheat_data, "Пшеница", 14)
# построим ряды относительных приращений
phosAgro_increments = calc_returns(phosAgro_data)
rusAgro_increments = calc_returns(rusAgro_data)
wheat_increments = calc_returns(wheat_data)
# гистограммы c характеристиками для приращений'
get_hist_stat([phosAgro_increments, rusAgro_increments, wheat_increments], ["приращения \"ФосАгро\"", "приращения \"Русагро\"", "приращеничя \"Пшеница\""])
# подбор распределений
sb.distplot(phosAgro_increments['OPEN'], fit=stats.t, label='ФосАгро')
plt.legend()
plt.figure()
sb.distplot(rusAgro_increments['OPEN'], fit=stats.t, label='РусАгро')
plt.legend()
plt.figure()
sb.distplot(wheat_increments['OPEN'], fit=stats.t, label='Пшеница')
plt.legend()
plt.grid()
plt.show()
# Модель ARIMA
phosAgro_data['OPEN'] = phosAgro_data['OPEN'].ewm(span=7, adjust=False).mean()
train_size = int(len(phosAgro_data['OPEN']) * 0.8)
train_data = phosAgro_data['OPEN'][:train_size]
test_data = phosAgro_data['OPEN'][train_size:]
# таким способом выдает константное либо линейное значение
#автоматическое
model = pm.auto_arima(train_data, trace=True)
# model_fit = model.fit(train_data)
# forecast = model_fit.predict(n_periods=len(test_data))
# print(forecast)
#вручную
mymodel = ARIMA(train_data, order=(1, 2, 0))
modelfit = mymodel.fit()
forecast = modelfit.predict(start=train_size, end=len(phosAgro_data) - 1)
# forecast = modelfit.predict()
print(forecast)
print(len(train_data))
print(len(forecast))
print(len(phosAgro_data['OPEN'][train_size:]))
plt.plot(phosAgro_data.index[train_size:], phosAgro_data['OPEN'][train_size:], label='real data')
plt.plot(phosAgro_data.index[train_size:], forecast, label='predict_data')
# plt.plot(phosAgro_data.index[1000:train_size], phosAgro_data['OPEN'][1000:train_size], label='real data')
# plt.plot(phosAgro_data.index[1000:train_size], forecast[1000:train_size], label='predict_data')
# wheat_increments['OPEN'] = wheat_increments['OPEN'].ewm(span=7, adjust=False).mean()
# train_size = int(len(wheat_increments['OPEN']) * 0.8)
# train_data = wheat_increments['OPEN'][:400]
# test_data = wheat_increments['OPEN'][400:425]
# model = pm.auto_arima(train_data, trace=True)
# mymodel = ARIMA(train_data.values, order=(2, 1, 2))
# mymodel = mymodel.fit()
# forecast = mymodel.predict(start=400, end=424)
# print(forecast)
# plt.plot(wheat_increments.index[400:425], wheat_increments['OPEN'][400:425], label='real data')
# plt.plot(wheat_increments.index[400:425], forecast, label='predict_data')
plt.legend()
plt.show()
