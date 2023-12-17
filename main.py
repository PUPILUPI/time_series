import csv
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from scipy import stats, signal
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import pmdarima as pm

#гарч модель
def print_info_garch(data, data_diff, name):
  train_data = data[0:-30]
  egarch_model  = arch_model(train_data, p=1,q=0 , vol ="EGARCH")
  model_fit = egarch_model.fit()
  model_fit.summary()
  rolling_predictions = []
  test_size = 365
  for i in range(test_size):
      train = data_diff[:-(test_size-i)]
      model = arch_model(train, p=1, q=1, vol ="EGARCH")
      model_fit = model.fit(disp='off')
      pred = model_fit.forecast(horizon=1)
      rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))
  plt.title(name)
  plt.plot(data_diff.index[-test_size:], data_diff[-test_size:], label ='оригинал')
  plt.plot(data_diff.index[-test_size:], rolling_predictions, color='r', label = 'прогноз')
  plt.grid()
  plt.legend()
  plt.show()


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
    data['OPEN'] = data['OPEN'].apply(lambda x: x / data['OPEN'][0])
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


#ф.наложения распределений
def show_distribution_density(data: pd.DataFrame, label: str):
    sb.distplot(data['OPEN'], fit=stats.t, label=label + '/стьюдента')
    plt.legend()
    plt.figure()
    # лог нормальное
    sb.distplot(data['OPEN'], fit=stats.lognorm, label=label + '/лог нормальное')
    plt.legend()
    plt.figure()
    # экспоненциальное
    sb.distplot(data['OPEN'], fit=stats.expon, label=label + '/экспоненциальное')
    plt.legend()
    plt.figure()
    # лапласа
    sb.distplot(data['OPEN'], fit=stats.laplace, label=label + '/лапласа')
    plt.legend()
    plt.figure()
    # нормальное
    sb.distplot(data['OPEN'], fit=stats.norm, label=label + '/нормальное')
    plt.legend()
    plt.figure()


# функция которая находит ряд приращений
def calc_returns(data: pd.DataFrame):
    returns = (data / data.shift(1)) - 1
    returns = returns.dropna()
    return returns


def get_autocorr(data : pd.DataFrame, name : str): # функция построения графика автокорреляции
    plot_acf(data, lags=50)
    plt.title(f'Автокорреляция {name}')


def get_part_autocorr(data : pd.DataFrame, name : str):  # функция построения графика частной автокорреляции
    plot_pacf(data, lags=50)
    plt.title(f'Частная автокорреляция {name}')


# подготовка данных для фосагро
phosAgro_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Вуз\\Ростовский\\PhosArgo.csv", ";");
convert_data(phosAgro_data)
phosAgro_data = convert_value(phosAgro_data)
phosAgro_data = phosAgro_data.set_index('DATE')
# для русагро
rusAgro_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Вуз\\Ростовский\\RusAgro.csv", ";");
convert_data(rusAgro_data)
convert_value(rusAgro_data)
rusAgro_data = rusAgro_data.set_index('DATE')
# для пшеницы
wheat_data = import_data("C:\\Users\\Xiaomi\\Desktop\\Вуз\\Ростовский\\wheat.csv", ";");
convert_data(wheat_data)
convert_value(wheat_data)
wheat_data = wheat_data.set_index('DATE')

# Строим графики
# plt.plot(phosAgro_data, label='ФосАгро')
# plt.plot(rusAgro_data, label='РусАгро')
# plt.plot(wheat_data, label='Пшеница')
# plt.legend()
# plt.show()
# вычисляем стандартные характеристики для рядов:
# Максимум и минимум,среднее, дисперсию, коэффициент ассиметрии, коэффициент эксцесса
# и строим их.
# get_hist_stat([phosAgro_data, rusAgro_data, wheat_data], ["ФосАгро", "Русагро", "Пшеница"])
# накладываем тренды
# fig2 = plt.figure()
# Показываем тренды
# show_trends(phosAgro_data, "ФосАгро", 3)
# show_trends(rusAgro_data, "РусАгро", 6)
# show_trends(wheat_data, "Пшеница", 14)
# подбор распределений
# show_distribution_density(rusAgro_data, 'РусАгро')

#АВТОКОРЕЛЛЯЦИЯ И ЧАСТИЧНАЯ АВТОКОРЕЛЛЯЦИЯ
# get_autocorr(rusAgro_data['OPEN'], 'Акции РусАгро')
# get_part_autocorr(rusAgro_data['OPEN'], 'Акции РусАгро')
# get_autocorr(phosAgro_data['OPEN'], 'Акции ФосАгро')
# get_part_autocorr(phosAgro_data['OPEN'], 'Акции ФосАгро')
# get_autocorr(wheat_data['OPEN'], 'Акции Пшеницы')
# get_part_autocorr(wheat_data['OPEN'], 'Акции Пшеницы')
#ПЕРИОДОГРАММЫ
# f, Pxx_spec = signal.periodogram(rusAgro_data['OPEN']) # возвращает значения частоты и спектр(мощность)
# plt.figure(figsize=(12, 6), dpi=120)
# # plt.semilogy(f, Pxx_spec)
# plt.stem(1/f, Pxx_spec) # по горизонтали периоды, по вертикали спектр
# plt.title('Периодограмма акций РусАгро')
# f, Pxx_spec = signal.periodogram(phosAgro_data['OPEN']) # возвращает значения частоты и спектр(мощность)
# plt.figure(figsize=(12, 6), dpi=120)
# # plt.semilogy(f, Pxx_spec)
# plt.stem(1/f, Pxx_spec) # по горизонтали периоды, по вертикали спектр
# plt.title('Периодограмма акций фосАгро')
# f, Pxx_spec = signal.periodogram(wheat_data['OPEN']) # возвращает значения частоты и спектр(мощность)
# plt.figure(figsize=(12, 6), dpi=120)
# # plt.semilogy(f, Pxx_spec)
# plt.stem(1/f, Pxx_spec)
# plt.title('Периодограмма акций Пшеница')
# plt.show()


#СТРОИМ РЯДЫ ОТНОСИТЕЛЬНЫХ ПРИРАЩЕНИЙ
phosAgro_increments = calc_returns(phosAgro_data)
rusAgro_increments = calc_returns(rusAgro_data)
wheat_increments = calc_returns(wheat_data)
# plt.plot(phosAgro_increments, label='отн. приращения "ФосАгро"')
# plt.legend()
# plt.figure()
# plt.plot(rusAgro_increments, label='отн. приращения "РусАгро"')
# plt.legend()
# plt.figure()
# plt.plot(wheat_increments, label='отн. приращения "Пшеница"')
# plt.legend()
# plt.show()

# гистограммы c характеристиками для приращений'
# get_hist_stat([phosAgro_increments, rusAgro_increments, wheat_increments], ["приращения \"ФосАгро\"", "приращения \"Русагро\"", "приращения \"Пшеница\""])
# show_distribution_density(phosAgro_increments, 'ФосАгро')
# show_distribution_density(rusAgro_increments, 'РусАгро')
# show_distribution_density(wheat_increments, 'Пшеница')

# Модель ARIMA

rusAgro_data = rusAgro_data.diff().dropna()
# phosAgro_data['OPEN'] = phosAgro_data['OPEN'].ewm(span=7, adjust=False).mean()
train_size = int(len(rusAgro_data['OPEN']) * 0.8)
train_data = rusAgro_data['OPEN'][:train_size]
test_data = rusAgro_data['OPEN'][train_size:]
# train_size = -20
# train_data = rusAgro_data['OPEN'][-100:train_size]
# test_data = rusAgro_data['OPEN'][train_size:]
# таким способом выдает константное либо линейное значение
#автоматическое

# model = pm.auto_arima(train_data, trace=True)
# model = pm.auto_arima(rusAgro_data['OPEN'], start_p=1, start_q=1,
#                       max_p=10, max_q=10,
#                       d=1, trace=True,
#                       seasonal=False)
# model_fit = model.fit(train_data)
# forecast = model_fit.predict(n_periods=len(test_data))
# print(forecast)
#вручную
plt.figure()
mymodel = ARIMA(train_data, order=(9, 1, 2))
modelfit = mymodel.fit()
# forecast = modelfit.predict(start=train_size, end=len(phosAgro_data) - 1)
forecast = modelfit.forecast(len(test_data))
rusAgro_data['forecast'] = [None]*(len(rusAgro_data)-len(test_data)) + list(forecast)
# forecast = modelfit.predict()
# print(forecast)
# print(len(train_data))
# plt.plot(phosAgro_data['OPEN'], label='real data')
# plt.plot(phosAgro_data['forecast'], label='forecast')

plt.plot(rusAgro_data.index[train_size:], np.cumsum(rusAgro_data['OPEN'][train_size:]), label='real data русАгро')
plt.plot(rusAgro_data.index[train_size:], np.cumsum(forecast), label='predict_data')
# plt.plot(phosAgro_data.index[1000:train_size], phosAgro_data['OPEN'][1000:train_size], label='real data')
# plt.plot(phosAgro_data.index[1000:train_size], forecast[1000:train_size], label='predict_data')
plt.legend()
plt.show()
