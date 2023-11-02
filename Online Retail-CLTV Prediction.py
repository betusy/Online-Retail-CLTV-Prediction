## CLTV Prediction with BG-NBD, Gamma Gamma

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_excel('/Users/betulyilmaz/Documents/Miuul/CRM Analytics/Bonus-Online Retail/online_retail_II.xlsx')
df = df_.copy()

df.head()
df.info()
df.describe().T

# Missing valuelari siliyoruz
df.isnull().sum()
df.dropna(inplace=True)

# Faturalardaki ‘C’ iptal edilen islemleri gosteriyor.
# İptal edilen işlemleri veri setinden cikariyoruz.
df = df[~df['Invoice'].astype(str).str.contains('C')]

# Aykiri degerleri baskiliyoruz.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

columns = ['Quantity', 'Price']
for col in columns:
    replace_with_thresholds(df, col)

# Total Price'i hesapliyoruz.
df['TotalPrice'] = df['Quantity'] * df['Price']

# Datasetteki max tarihi buluyoruz.
# Bu tarihten 2 gun sonrayi analiz tarihi olarak kabul ediyoruz.
df['InvoiceDate'].max() #2010-12-09
today_date = dt.datetime(2010, 12, 11)

# Lifetime veri yapisinin hazirlanmasi

# recency: son satin alma uzerinden gecen zaman. haftalik. (kullanici ozelinde)
# T: musterinin yasi. haftalik. (analiz tarihinden ne kadar sure once ilk satin alma yapilmis)
# frequency: tekrar eden toplam satin alma sayisi (frequency > 1)
# monetary_value: satin alma basina ortalama kazanc.

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days, #recency
                                                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days], #T
                                        'Invoice': lambda Invoice: Invoice.nunique(), #frequency
                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()}) #monetary

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df['recency'] = cltv_df['recency'] / 7
cltv_df["T"] = cltv_df["T"] / 7

## BG-NBD Modeli
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# 1 hafta icinde en cok satin alma bekledigimiz 10 musteri
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False)

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10) # yukaridaki fonk ile aynidir. predict gama gama icin gecerli degildir.

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# 1 ay icinde en cok satin alma bekledigimiz 10 musteri

bgf.predict(4, cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# 3 ayda tum sirketin beklenen satis sayisi

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

# tahmin sonuclarinin degerlendirilmesi

plot_period_transactions(bgf)
plt.show()

## Gamma-Gamma Modelinin Kurulmasi

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values('expected_average_profit', ascending=False).head(10)

## BG-NBD ve GG modeli ile CLTV'nin hesaplanmasi

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3, # 3 aylik
                                   freq='W', # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv.head()
cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on='Customer ID', how='left')
cltv_final.sort_values(by='clv', ascending=False).head(10)

# CLTV'ye gore segmentler olusturulmasi

cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels=["D", "C", "B", "A"])
cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})