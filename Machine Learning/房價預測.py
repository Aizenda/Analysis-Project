import pandas as pd
import os
import matplotlib as mpl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt
import re
#顯示中文
mpl.rc('font', family='Microsoft JhengHei')
time=datetime.datetime.now().year

location_str = """
台北市 A 苗栗縣 K 花蓮縣 U
台中市 B 台中縣 L 台東縣 V
基隆市 C 南投縣 M 澎湖縣 X
台南市 D 彰化縣 N 陽明山 Y
高雄市 E 雲林縣 P 金門縣 W
台北縣 F 嘉義縣 Q 連江縣 Z
宜蘭縣 G 台南縣 R 嘉義市 I
桃園縣 H 高雄縣 S 新竹市 O
新竹縣 J 屏東縣 T
"""

# 將 location_str 字符串分割成縣市名稱和其對應的英文字母，然後轉換為字典形式
location_dict = dict(zip(location_str.split()[::2], location_str.lower().split()[1::2]))

# 測試輸入"台北市"獲取其對應的字母
county=input(str('請輸入地址:'))
result = location_dict.get(county[:3])

# 歷年資料夾
path=os.chdir('E:\\python_學習資料\\專案數據\\實價登入\\解壓縮資料夾\\')
dirs = [d for d in os.listdir(path) if d[:4] == 'real']
dfs = []
for d in dirs:
    df = pd.read_csv(os.path.join(d,result + '_lvr_land_a.csv'), index_col=False)
    dfs.append(df.iloc[1:])
df = pd.concat(dfs, sort=True)
#檢查缺失值
df.info()
import matplotlib.pyplot as plt
import missingno as msno

msno.matrix(df)
plt.show()
plt.subplots(figsize=(15,5)) # 設定畫面大小
df.isnull().sum().plot.bar()
plt.show()

#處理空值
df.isnull().sum()
#建材空值為土地不含房產，故刪除
df = df[df['主要建材'].notnull()]
#主要用途未填寫，會引響模型預測
df = df[df['主要用途'].notnull()]
#備註13項，多為不正常交易，故刪除含有備註行
df = df[df['備註'].isnull()]
#單價平方公尺可以計算，以計算值取代
df['單價元平方公尺']=(df['總價元'].astype(float)-df['車位總價元'].astype(float))/(df['建物移轉總面積平方公尺'].astype(float)-df['車位移轉總面積平方公尺'].astype(float))
df=df[df['單價元平方公尺'].notnull()]
#用屋齡取代建築完成年
df = df[(df['建築完成年月'].str.len() == 7)]
pat = '(00).+'
df = df[df['建築完成年月'].str.contains(pat) == 00]
df['建築完成年(民國)']= df['建築完成年月'].str[:3]
df['屋齡'] = (time-1911)-(pd.to_numeric(df['建築完成年(民國)'], errors='coerce')).astype(float)
df = df.drop(['建築完成年月','建築完成年(民國)'],axis=1)
#總樓層數缺失值數量少於1%，直接刪除引響不大
df = df[df['總樓層數'].notnull()]
#將車位類改為有無車位
df['有無車位'] = df['車位類別'].str.len()>1
df['有無車位'] = df['有無車位'].replace(True,'有')
df['有無車位'] = df['有無車位'].replace(False,'無')
df = df.drop('車位類別',axis=1)
#都市使用分區缺失少於1%，刪除
df = df[df['都市土地使用分區'].notnull()]
#電梯，缺失值過多，採用補值
df['電梯'].fillna(value='有', inplace=True)
#篩選出包含住宅區的欄位
df = df[ (df['主要用途'].str[:1] == '住')]
#剔除單純車位或土地
df = df[ (df['交易標的'].str[:2] == '房地') | (df['交易標的'].str[:2] == '建物')]
#將交易年月日改為交易年，減少複雜度
df['交易年'] = (df['交易年月日'].str[:3]).astype(float)
df = df.drop(['交易年月日'],axis=1)
#將數字數據，轉為float，便後續處理
df[['土地移轉總面積平方公尺','建物現況格局-廳','建物現況格局-房','建物現況格局-衛','建物移轉總面積平方公尺','總價元','車位移轉總面積平方公尺','車位總價元','附屬建物面積','陽台面積']]=df[['土地移轉總面積平方公尺','建物現況格局-廳','建物現況格局-房','建物現況格局-衛','建物移轉總面積平方公尺','總價元','車位移轉總面積平方公尺','車位總價元','附屬建物面積','陽台面積']].astype(float)
df[['主建物面積']]=df[['主建物面積']].astype(float)



df['主建物占比']=df['主建物面積']/df['建物移轉總面積平方公尺']*100
# 選擇感興趣的連續型變數，這裡只是示例，請根據您的數據選擇相應欄位
continuous_variables = ['主建物面積',
'單價元平方公尺',
'土地移轉總面積平方公尺',
'建物現況格局-廳',
'建物現況格局-房',
'建物現況格局-衛',
'建物移轉總面積平方公尺',
'建築完成年月',
'總價元',
'車位移轉總面積平方公尺',
'車位總價元',
'附屬建物面積',
'陽台面積']

# 創建包含連續型變數的子DataFrame
df_continuous = df[continuous_variables]

# 添加總價元列
df_continuous['總價元'] = df['總價元']

# 使用Seaborn創建散點矩陣
sns.set(style="ticks")
sns.pairplot(df_continuous, diag_kind="kde")
plt.show()

#刪除缺失值過多且不具引響預測結果的欄位
df = df.drop(['移轉編號','編號','非都市土地使用分區','非都市土地使用編定','備註','土地位置建物門牌'
              ,'土地移轉總面積平方公尺','單價元平方公尺','主要建材','交易標的','交易筆棟數','建物現況格局-隔間'
              ,'附屬建物面積','陽台面積','主要用途','車位移轉總面積平方公尺','車位總價元','都市土地使用分區'],axis=1)
df['建物型態'] = df['建物型態'].str.split('(').str[0]
df['移轉層次']=df['移轉層次'].str.split('，').str[0]


X_continuous = df.select_dtypes(exclude=["object"])

X_onehot = df.drop(df.select_dtypes(exclude=["object"]), axis=1) 
print(df.columns)
plt.scatter(df['主建物面積'], df['總價元'])
plt.xlabel('主建物面積')
plt.ylabel('總價元')
plt.title('主建物面積與總價元的關係')
plt.show()
import numpy as np
#特徵工程
for column in X_onehot.columns:
    le = LabelEncoder()
    X_onehot[column] = le.fit_transform(X_onehot[column])


# # 將降維後的資料和連續型資料合併
X = pd.concat((X_onehot, X_continuous.drop(['總價元','主建物面積'],axis=1)), axis=1)

y = df['總價元']

#分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)
#測試資料的預測結果
y_pred = regressor.predict(X_test)
#評估模型效力
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# 顯示評估結果
print('*'*50)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)







