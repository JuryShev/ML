import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import lag_plot
import numpy as np
import calendar
train = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/train.csv', nrows=100000)
train_drop=train
keep_id = train_drop[train_drop.type == "Assessment"]['installation_id'].drop_duplicates()
train = pd.merge(train, keep_id, on="installation_id", how="inner")

print(train.shape[0])
keep_id.shape

print("OK")


plt.show()
#delta_time= \
#day=train[['timestamp']].copy()
day=pd.DataFrame()
Session_time=pd.DataFrame({"game_session":[0], "long_time":[0], "ID_num0":[0], "tipe":[0] })



#________ужать_______________
day[['date','time']] = train['timestamp'].str.split('T',expand=True)
day[['hour','minut', 'sec']]=day['time'].str.split(':',expand=True)
day[['hour']]=day[['hour']].astype(int)

#*60+day.iloc[[3],[2]]
ser_data=pd.Series(day['date'], index=day.index)
ser_hour=pd.Series(day['hour'], index=day.index)
ser_minut=pd.Series(day['minut'], index=day.index)
#_____________delta_time____________________________________________________________
delta_time=pd.to_numeric(ser_hour[1])*60+pd.to_numeric(ser_minut[1])
delta_time=pd.to_numeric(ser_hour[200])*60+pd.to_numeric(ser_minut[200])-delta_time
#___________________________________________________________________________________
#2b9d5af79bcdb79f
num0=0
num1=0
num_z=0
Session_time.loc[[0], ['game_session']] =train.iloc[num0, 1]
Session_time.loc[[0], ['ID_num0']] =num0
Session_time.loc[[num1], ['tipe']]=train.iloc[0, 9]

#train.loc[[num0],['game_session']]==Session_time.loc[[0], ['game_session']]
while(num0<(train.shape[0]-1)):
    delta_time0 = pd.to_numeric(ser_hour[num0]) * 60 + pd.to_numeric(ser_minut[num0])

    while(Session_time.iloc[num1, 0]==train.iloc[num0, 1] and num0<(train.shape[0]-1)):
        num0=num0+1
        # 1 Записать время в long_time
        # 2 Прописать дельту
        # 3 расчитать время сезона

    delta_time = pd.to_numeric(ser_hour[num0-1]) * 60 + pd.to_numeric(ser_minut[num0-1]) - delta_time0
    if(delta_time<0):
        delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(ser_minut[num0 - 1])+1440 - delta_time0
    if(delta_time<1):
        num_z=num_z+1
        #train = pd.merge(train, train.loc[[num0], ['game_session']], on="game_session", how="inner")

    print(num0)
    Session_time.loc[[num1], ['long_time']] = delta_time
    Session_time = Session_time.append({'game_session': train.iloc[num0, 1]}, ignore_index=True)
    num1=num1+1
    Session_time.loc[[num1], ['ID_num0']] = num0
    Session_time.loc[[num1], ['tipe']]=train.iloc[num0, 9]

keep_ltime=Session_time[Session_time.long_time >0]['game_session'].drop_duplicates()
print(keep_ltime.dtype)

train = pd.merge(train, keep_ltime,  on="game_session", how="inner")
Session_time=pd.DataFrame({"game_session":[0], "long_time":[0], "ID_num0":[0], "tipe":[0] })
#____________________________________________________________________________

num0=0
num1=0
num_z=0
Session_time.loc[[0], ['game_session']] =train.iloc[num0, 1]
Session_time.loc[[0], ['ID_num0']] =num0
Session_time.loc[[num1], ['tipe']]=train.iloc[0, 9]

#train.loc[[num0],['game_session']]==Session_time.loc[[0], ['game_session']]
while(num0<(train.shape[0]-1)):
    delta_time0 = pd.to_numeric(ser_hour[num0]) * 60 + pd.to_numeric(ser_minut[num0])

    while(Session_time.iloc[num1, 0]==train.iloc[num0, 1] and num0<(train.shape[0]-1)):
        num0=num0+1
        # 1 Записать время в long_time
        # 2 Прописать дельту
        # 3 расчитать время сезона

    delta_time = pd.to_numeric(ser_hour[num0-1]) * 60 + pd.to_numeric(ser_minut[num0-1]) - delta_time0
    if(delta_time<0):
        delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(ser_minut[num0 - 1])+1440 - delta_time0
    if(delta_time<1):
        num_z=num_z+1
        #train = pd.merge(train, train.loc[[num0], ['game_session']], on="game_session", how="inner")

    print(num0)
    Session_time.loc[[num1], ['long_time']] = delta_time
    Session_time = Session_time.append({'game_session': train.iloc[num0, 1]}, ignore_index=True)
    num1=num1+1
    Session_time.loc[[num1], ['ID_num0']] = num0
    Session_time.loc[[num1], ['tipe']]=train.iloc[num0, 9]
#____________________________________________________________________________
Session_time=Session_time.sort_values(by=['long_time'],  ascending=False)
ser_data = pd.to_datetime(ser_data)
#ser_time = pd.to_datetime(ser_time)
fig = plt.figure(figsize=(12, 10))
day_m=ser_data.dt.dayofweek

#se = train.groupby('dayofweek')['dayofweek'].count()
#se.index = list(calendar.day_abbr)
day_m.plot.hist()
day_m=day_m.value_counts()
plt.title("Event counts by day of week")
plt.xticks(rotation=0)
plt.show()

lag_plot(ser_hour)
plt.show()

ser_hour=ser_hour.value_counts().sort_index()
ser_hour.plot.bar()
plt.show()

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
ax1 = sns.countplot(y="type", data=train, color="blue", order = train.type.value_counts().index)
plt.title("number of events by type")

ax2 = fig.add_subplot(212)
ax2 = sns.countplot(y="world", data=train, color="blue", order = train.world.value_counts().index)
plt.title("number of events by world")

plt.tight_layout(pad=0)
plt.show()


plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(12,10))
se = train.title.value_counts().sort_values(ascending=True)
se.plot.barh()
plt.title("Event counts by title")
plt.xticks(rotation=0)