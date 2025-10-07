# --------------------------- Lab_work_1  ------------------------------------

'''

Виконала: Швецова Анна
Lab_work_1, варіант 10, ІІ рівень складності (8 балів):
Комбінації:
 - рівномірний шум + кубічний тренд
 - нормальний шум + лінійний тренд
Реальні дані – 3 показники ("Купівля", "Продаж", "КурсНБУ") з Oschadbank (USD).xls

'''

import numpy as np
import pandas as pd
import math as mt
from matplotlib import pyplot as plt



# ------------------------ ФУНКЦІЯ парсингу реальних даних --------------------------

def file_parsing (URL, File_name, Data_name):

    '''

    :param URL: адреса сайту для парсингу str
    :param File_name:
    :param Data_name:
    :return:
    '''

    d = pd.read_excel(File_name)
    for name, values in d[[Data_name]].items():
    # for name, values in d[[Data_name]].iteritems(): # приклад оновлення версій pandas для директиви iteritems
        print(values)
    S_real = np.zeros((len(values)))
    for i in range(len(values)):
        S_real[i] = values[i]
    print('Джерело даних: ', URL)
    return S_real


# ---------------------- ФУНКЦІЇ тестової аддитивної моделі -------------------------

# ----------- рівномірний закон розводілу номерів АВ в межах вибірки ----------------
def randomAM (n):

    '''

    :param n: кількість реалізацій ВВ - об'єм вибірки
    :return: номери АВ
    '''

    SAV = np.zeros((nAV))
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(0, iter)  # параметри закону задаются межами аргументу
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    # -------------- генерація номерів АВ за рівномірним законом  -------------------
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))  # рівномірний розкид номерів АВ в межах вибірки розміром 0-iter
    print('номери АВ: SAV=', SAV)
    print('----- статистичны характеристики РІВНОМІРНОГО закону розподілу ВВ -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return SAV

# ------------------------- нормальний закон розводілу ВВ ----------------------------
def randoNORM (dm, dsig, iter):

    '''

    :param dm:
    :param dsig:
    :param iter:
    :return:
    '''

    S = np.random.normal(dm, dsig, iter)  # нормальний закон розподілу ВВ з вибіркою єбємом iter та параметрами: dm, dsig
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- статистичны характеристики НОРМАЛЬНОЇ похибки вимірів -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('------------------------------------------------------------------')
    # гістограма закону розподілу ВВ
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S

# ------------------- модель ідеального тренду (квадратичний закон)  ------------------
def Model (n):

    '''

    :param n:
    :return:
    '''

    S0=np.zeros((n))
    for i in range(n):
        S0[i]=(0.0000005*i*i)    # квадратична модель реального процесу
        # S0[i] = 45  # квадратична модель реального процесу
    return S0

# ---------------- модель виміру (квадратичний закон) з нормальний шумом ---------------
def Model_NORM (SN, S0N, n):

    '''

    :param SN:
    :param S0N:
    :param n:
    :return:
    '''

    SV=np.zeros((n))
    for i in range(n):
        SV[i] = S0N[i]+SN[i]
    return SV

# ----- модель виміру (квадратичний закон) з нормальний шумом + АНОМАЛЬНІ ВИМІРИ
def Model_NORM_AV (S0, SV, nAV, Q_AV):

    '''

    :param S0:
    :param SV:
    :param nAV:
    :param Q_AV:
    :return:
    '''


    SV_AV = SV
    SSAV = np.random.normal(dm, (Q_AV * dsig), nAV)  # аномальна випадкова похибка з нормальним законом
    for i in range(nAV):
        k=int (SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]        # аномальні вимірів з рівномірно розподіленими номерами
    return SV_AV


def Stat_characteristics (SL, Text):

    '''

    :param SL:
    :param Text:
    :return:
    '''

    # статистичні характеристики вибірки з урахуванням тренду за МНК
    def Trend_MLS (SL):
        iter = len(SL)
        Yout = MNK_Stat_characteristics(SL)  # визначається за МНК
        SL0 = np.zeros((iter))
        for i in range(iter):
            SL0[i] = SL[i] - Yout[i, 0]
        return SL0

    # статистичні характеристики вибірки з урахуванням тренду за вихідними даними
    def Trend_Сonstant(SL):
        iter = len(SL)
        Yout = Model(iter)
        SL0 = np.zeros((iter))
        for i in range(iter):
            SL0[i] = SL[i] - Yout[i]
        return SL0

    SL0 = Trend_MLS (SL)        # статистичні характеристики вибірки з урахуванням тренду за МНК

    # SL0 = Trend_Сonstant(SL)    # статистичні характеристики вибірки з урахуванням тренду за вихідними даними

    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text ,'-------------')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')
    return

# ------------- МНК згладжуваннядля визначення стат. характеристик -------------
def MNK_Stat_characteristics (S0):

    '''

    :param S0:
    :return:
    '''

    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------
def Plot_AV (S0_L, SV_L, Text):

    '''

    :param S0_L:
    :param SV_L:
    :param Text:
    :return:
    '''

    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return

# ---------------------------- нові тренди ---------------------------------------
def trend_linear(t, a0, a1):
    # y = a0 + a1 * t
    return a0 + a1 * t

def trend_cubic(t, a0, a1, a2, a3):
    # y = a0 + a1*t + a2*t^2 + a3*t^3
    return a0 + a1*t + a2*(t**2) + a3*(t**3)

# --------------------- Генерація рівномірної похибки (U[a,b]) -------------------------
def randomUNIF(a, b, n):
    S = np.random.uniform(a, b, n)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('----- статистичні характеристики РІВНОМІРНОЇ похибки -----')
    print('математичне сподівання (медіана) =', mS)
    print('дисперсія =', dS)
    print('СКВ =', scvS)
    print('----------------------------------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return S

# ------------------------ МНК-оцінка поліномом потрібного степеня + R^2 (лекція 3) --------------------
def fit_poly_and_r2(t, y, degree):
    # NumPy polyfit/polyval реалізує МНК; R^2 визначаємо як 1 - SS_res/SS_tot
    coef = np.polyfit(t, y, deg=degree)
    y_hat = np.polyval(coef, t)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0
    return coef, y_hat, r2

# --------------------------- Лінійка з трьома кривими: виміри, ідеальний тренд, МНК-оцінка ----------------------
def plot_timeseries3(t, y, y_trend, y_hat, title):
    plt.clf()
    plt.plot(y, label='виміряні дані')
    plt.plot(y_trend, label='ідеальний тренд')
    plt.plot(y_hat, label='оцінка МНК')
    plt.ylabel(title)
    plt.legend()
    plt.show()

# -------------------------------- БЛОК ГОЛОВНИХ ВИКЛИКІВ ----------------------------------

if __name__ == '__main__':

    # ------------------------------ сегмент констант ---------------------------------------
    n = 10000             # кількість реалізацій
    t = np.arange(n)      # вісь часу
    dm = 0.0              # параметри нормального закону
    dsig = 1.0            # (середнє, СКВ)
    ua, ub = -2.0, 2.0    # межі рівномірної похибки U[a,b]

    # ------------------------------ сегмент даних -------------------------------------------
    # ------------ дві комбінації для варіанта 10: U+кубічний, N+лінійний -------------------

    # -------------- Комбінація 1: РІВНОМІРНИЙ шум + КУБІЧНИЙ тренд -------------
    S0_cubic = trend_cubic(t, a0=10.0, a1=0.3, a2=-0.005, a3=0.00002)  # ідеальний кубічний тренд
    eps_U    = np.random.uniform(ua, ub, size=n)                        # рівномірний шум U[a,b]
    SV_U     = S0_cubic + eps_U                                         # адитивна модель
    # візуалізація та статистики
    Plot_AV(S0_cubic, SV_U, 'кубічний тренд + Рівномірний шум')
    Stat_characteristics(SV_U, 'Вибірка (U + cubic)')

    # -------------- Комбінація 2: НОРМАЛЬНИЙ шум + ЛІНІЙНИЙ тренд ---------------
    S0_lin = trend_linear(t, a0=5.0, a1=0.1)                            # ідеальний лінійний тренд
    S_N    = randoNORM(dm, dsig, n)                                     # нормальний шум (твоя функція)
    SV_N   = S0_lin + S_N                                                # адитивна модель

    Plot_AV(S0_lin, SV_N, 'лінійний тренд + Нормальний шум')
    Stat_characteristics(SV_N, 'Вибірка (N + linear)')
    

    # -------------------------------- Реальні дані -------------------------------------------
try:
    S_buy  = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'Купівля')
    S_sell = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'Продаж')
    S_nbu  = file_parsing('https://www.oschadbank.ua/rates-archive', 'Oschadbank (USD).xls', 'КурсНбу')

    # для кожного реального ряду — покажемо ряд і просту лінійну апроксимацію
    for name, Sreal in [('Купівля', S_buy), ('Продаж', S_sell), ('КурсНбу', S_nbu)]:
        tt = np.arange(len(Sreal))
        # МНК-оцінка лінійним поліномом
        coef = np.polyfit(tt, Sreal, deg=1)
        yhat = np.polyval(coef, tt)

        # графік/статистика:
        Plot_AV(yhat, Sreal, f'{name}: реальні дані + лінійна апроксимація')
        Stat_characteristics(Sreal, f'{name}: реальні дані')

except Exception as e:
    print("\n[Попередження] Реальні дані пропущено:", e)
