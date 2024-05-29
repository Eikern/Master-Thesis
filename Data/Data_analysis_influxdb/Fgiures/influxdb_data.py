from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns

#Username,password,IP and hostname in the client's https request to the influxdb server has been removed for security reasons.

ip = ''
host ='' 


start_year = 2024
start_month = 4
start_day = 1

end_year = 2024
end_month = 5   
end_day =1


start_of_month = pd.Timestamp(start_year,start_month, start_day)
end_of_month = pd.Timestamp(end_year, end_month, end_day)

dashboardTime =start_of_month
upperDashboardTime = end_of_month

time_series = pd.date_range(start=start_of_month, end=end_of_month, freq='T')



client = InfluxDBClient(host=ip, port=8086, username='', password='', database='')

query_solar_prod = f'SELECT abs(last("value")) AS "solar_prod" FROM "homeassistantdb"."autogen"."W" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'03_solarinput63a_active_power\' GROUP BY time(1m) FILL(previous)'
query_varmepumpe = f'SELECT abs(last("value")) AS "varmepumpe" FROM "homeassistantdb"."autogen"."W" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'11_varmepumpe32a_active_power\' GROUP BY time(1m) FILL(previous)'
query_elbil =  f'SELECT abs(last("value")) AS "elbil" FROM "homeassistantdb"."autogen"."W" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'u05_billader16a_active_power\' GROUP BY time(1m) FILL(previous)'
query_import = f'SELECT abs(last("value")) AS "import" FROM "homeassistantdb"."autogen"."W" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'ams_linje6_p\' GROUP BY time(1m) FILL(previous)'
query_export = f'SELECT last("value") AS "export" FROM "homeassistantdb"."autogen"."W" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'ams_linje6_po\' GROUP BY time(1m) FILL(previous)'


result_solar_prod = client.query(query_solar_prod)
result_varmepumpe = client.query(query_varmepumpe)
result_elbil = client.query(query_elbil)
result_import = client.query(query_import)
result_export = client.query(query_export)

df1 = pd.DataFrame(result_solar_prod.get_points())
df2 = pd.DataFrame(result_varmepumpe.get_points())
df3 = pd.DataFrame(result_elbil.get_points())
df4 = pd.DataFrame(result_import.get_points())
df5 = pd.DataFrame(result_export.get_points()) 

dataframes = [df1, df2, df3, df4, df5]

for i, df in enumerate(dataframes):
    if df.empty:
        zeros_df = pd.DataFrame(0, index=range(len(time_series)),columns=df.columns)
        dataframes[i] = zeros_df

df = pd.DataFrame()
df["time"] = time_series


for i, y in enumerate(dataframes):
        y = y.drop(columns='time') if 'time' in y.columns else y
        y = y.fillna(0)
        df = pd.merge(df, y, how='outer',left_index=True,right_index=True)

df.set_index("time", inplace = True)
df_prod = df.copy()

df_prod1 = df.copy().resample("15T").mean() 
df_prod2 = df.copy().resample("30T").mean() 
df_prod3 = df.copy().resample("H").mean()

start_time = time_series[0] + pd.Timedelta(hours=4)
end_time = start_time + pd.Timedelta(hours=1)

fig, (ax1, ax2, ax3, ax4,ax5) = plt.subplots(5, 1, figsize=(10, 12))

# Plot for the raw  data (1_min_mean values)
ax1.step(df_prod[start_time:end_time].index, df_prod.loc[start_time:end_time]["varmepumpe"], where='post')
ax1.set_title('Heat Pump Power Usage')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (W)')
ax1.grid()

# Plot for 15-minute average  data
ax2.step(df_prod1[start_time:end_time].index, df_prod1.loc[start_time:end_time]["varmepumpe"], where='post')
ax2.set_title('Average Power Every 15 Minutes')
ax2.set_xlabel('Time')
ax2.set_ylabel('Power (W)')
ax2.grid()

# Plot for 30-minute average  data
ax3.step(df_prod2[start_time:end_time].index, df_prod2.loc[start_time:end_time]["varmepumpe"], where='post')
ax3.set_title('Average Power Every 30 Minutes')
ax3.set_xlabel('Time')
ax3.set_ylabel('Power (W)')
ax3.grid()

# Plot for 1-hour average  data
ax4.step(df_prod3[start_time:end_time].index, df_prod3.loc[start_time:end_time]["varmepumpe"], where='post')
ax4.set_title('Average Power Every Hour')
ax4.set_xlabel('Time')
ax4.set_ylabel('Power (W)')
ax4.grid()

ax5.step(df_prod[start_time:end_time].index, df_prod.loc[start_time:end_time]["varmepumpe"], where='post')
ax5.step(df_prod1[start_time:end_time].index, df_prod1.loc[start_time:end_time]["varmepumpe"], where='post')
ax5.step(df_prod2[start_time:end_time].index, df_prod2.loc[start_time:end_time]["varmepumpe"], where='post')
ax5.step(df_prod3[start_time:end_time].index, df_prod3.loc[start_time:end_time]["varmepumpe"], where='post')
ax5.set_xlabel('time')
ax5.set_ylabel('Power (W)')
ax5.grid()

plt.tight_layout()
plt.show()
#%%
df_energy = df.copy()/60
df_prod11 = df.copy().resample("15T").sum()
df_prod22 = df.copy().resample("30T").sum() 
df_prod33 = df.copy().resample("H").sum()
df_prod11["energy_varmepumpe_15min"]=df_prod11["varmepumpe"]/15
df_prod22["energy_varmepumpe_30min"]=df_prod22["varmepumpe"]/30
df_prod33["energy_varmepumpe_1hour"]=df_prod33["varmepumpe"]/60


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))

# Plot of the original power data for the heat pump (converted to kWh by dividing by 60)
ax1.step(df_prod[start_time:end_time].index, df_prod.loc[start_time:end_time]["varmepumpe"], where='post')
ax1.set_title('Heat Pump Power Usage (W)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (kWh)')
ax1.grid(True)

# Plotting the 15-minute aggregated energy data
ax2.step(df_prod11[start_time:end_time].index, df_prod11.loc[start_time:end_time]["energy_varmepumpe_15min"], where='post')
ax2.set_title('Energy Consumption Every 15 Minutes')
ax2.set_xlabel('Time')
ax2.set_ylabel('Energy (Wh)')
ax2.grid(True)

# Plotting the 30-minute aggregated energy data
ax3.step(df_prod22[start_time:end_time].index, df_prod22.loc[start_time:end_time]["energy_varmepumpe_30min"], where='post')
ax3.set_title('Energy Consumption Every 30 Minutes')
ax3.set_xlabel('Time')
ax3.set_ylabel('Energy (Wh)')
ax3.grid(True)

# Plotting the 1-hour aggregated energy data
ax4.step(df_prod33[start_time:end_time].index, df_prod33.loc[start_time:end_time]["energy_varmepumpe_1hour"], where='post')
ax4.set_title('Energy Consumption Every Hour')
ax4.set_xlabel('Time')
ax4.set_ylabel('Energy (Wh)')
ax4.grid(True)

plt.tight_layout()
plt.show()
df_prod1['fixed_load'] =df_prod1['import']+(df_prod1['solar_prod']-df_prod1['export'])-df_prod1['varmepumpe']
df.plot()

for column in df.columns:
    fig, ax = plt.subplots()
    
    ax.plot(df.index, df[column])
    
    ax.set_xlabel('Time')
    ax.set_ylabel("Watt")
    ax.set_title(f'Plot of {column}')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
    plt.xticks(rotation=45)
    
    plt.show()

#Radiatorer

query_atrium_nord = f'SELECT last("value") AS "atrium_nord" FROM "homeassistantdb"."autogen"."%" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'fussbodenheizung_anbau_331_level_ch1\' GROUP BY time(1m) FILL(previous)'
query_anbau_bad = f'SELECT last("value") AS "anbau_bad" FROM "homeassistantdb"."autogen"."%" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'fussbodenheizung_anbau_331_level_ch2\' GROUP BY time(1m) FILL(previous)'
query_anbau_stua = f'SELECT last("value") AS "anbau_stua" FROM "homeassistantdb"."autogen"."%" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'fussbodenheizung_anbau_331_level_ch3\' GROUP BY time(1m) FILL(previous)'
query_anbau_syd = f'SELECT last("value") AS "anbau_syd" FROM "homeassistantdb"."autogen"."%" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'fussbodenheizung_anbau_331_level_ch4\' GROUP BY time(1m) FILL(previous)'

result_atrium_nord = client.query(query_atrium_nord)
result_anbau_bad = client.query(query_anbau_bad)
result_anbau_stua = client.query(query_anbau_stua)
result_anbau_syd = client.query(query_anbau_syd)

df1 = pd.DataFrame(result_atrium_nord.get_points())
df2 = pd.DataFrame(result_anbau_bad.get_points())
df3 = pd.DataFrame(result_anbau_stua.get_points())
df4 = pd.DataFrame(result_anbau_syd.get_points())

dataframes = [df1, df2, df3, df4]


df = pd.DataFrame()
df["time"] = time_series

for i, df in enumerate(dataframes):
    if df.empty:
        zeros_df = pd.DataFrame(0, index=range(len(time_series)),columns=df.columns)
        dataframes[i] = zeros_df

df = pd.DataFrame()
df["time"] = time_series


for i, y in enumerate(dataframes):
        y = y.drop(columns='time') if 'time' in y.columns else y
        y = y.fillna(0)
        df = pd.merge(df, y, how='outer',left_index=True,right_index=True)

df.set_index("time", inplace = True)
df_rad = df.copy()



df.plot()

for column in df.columns:
    fig, ax = plt.subplots()
    
    ax.plot(df.index, df[column])
    
    ax.set_xlabel('Time')
    ax.set_ylabel("%")  
    ax.set_title(f'Plot of {column}')
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    plt.show()

#Temp

query_radiator_stua_nord = f'SELECT mean("current_temperature") AS "radiator_stua_nord" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_stua_nord\' GROUP BY time(1m) FILL(previous)'
query_state_radiator_stua_nord = f'SELECT mean("temperature") AS "state_radiator_stua_nord" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_stua_nord\' GROUP BY time(1m) FILL(previous)'

query_radiator_stua_sued = f'SELECT mean("current_temperature") AS "radiator_stua_sued" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_stua_sued\' GROUP BY time(1m) FILL(previous)'
query_state_radiator_stua_sued = f'SELECT mean("temperature") AS "state_radiator_stua_sued" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_stua_sued\' GROUP BY time(1m) FILL(previous)'

query_radiator_joar = f'SELECT mean("current_temperature") AS "radiator_joar" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_joar\' GROUP BY time(1m) FILL(previous)'
query_state_radiator_joar = f'SELECT mean("temperature") AS "state_radiator_joar" FROM "homeassistantdb"."autogen"."state" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'radiator_joar\' GROUP BY time(1m) FILL(previous)'

query_utetemp =f'SELECT mean("value") AS "mean_value" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'temp_humidity_draussen_555_temperature\' GROUP BY time(1m) FILL(previous)'

result_radiator_stua_nord = client.query(query_radiator_stua_nord)
result_state_radiator_stua_nord = client.query(query_state_radiator_stua_nord)

result_radiator_stua_sued = client.query(query_radiator_stua_sued)
result_state_radiator_stua_sued = client.query(query_state_radiator_stua_sued)

result_radiator_joar = client.query(query_radiator_joar)
result_state_radiator_joar = client.query(query_state_radiator_joar)

result_utetemp= client.query(query_utetemp)


df1 = pd.DataFrame(result_radiator_stua_nord.get_points())
df2 = pd.DataFrame(result_state_radiator_stua_nord.get_points())

df3 = pd.DataFrame(result_radiator_stua_sued.get_points())
df4 = pd.DataFrame(result_state_radiator_stua_sued.get_points())

df5 = pd.DataFrame(result_radiator_joar.get_points())
df6 = pd.DataFrame(result_state_radiator_joar.get_points())

df7 = pd.DataFrame(result_utetemp.get_points())

dataframes = [df1, df2, df3, df4, df5, df6,df7]

df = pd.DataFrame()
df["time"] = time_series
#df7["time"] = time_series[:-1]


for i, df in enumerate(dataframes):
    if df.empty:
       
        zeros_df = pd.DataFrame(0, index=range(len(time_series)),columns=df.columns)
        dataframes[i] = zeros_df

df = pd.DataFrame()
df["time"] = time_series

for i, y in enumerate(dataframes):
        y = y.drop(columns='time') if 'time' in y.columns else y
        y = y.fillna(0)
        df = pd.merge(df, y, how='outer',left_index=True,right_index=True)

df.set_index("time", inplace = True)
df7.set_index("time", inplace = True)
df7.to_csv("mean_value_data.tsv", sep='\t', index=True, header=None)
df71 = df7.copy()
df_temp = df.copy().resample("15T").mean()



for column in df_temp.columns:
    file_name = f"{column}_data.tsv"
    #new_index = ['t' + str(s) for s in range(0, len(df))]
    #df_temp.index = new_index
    #df_temp.ffill(inplace=True)
    df_temp[[column]].to_csv(file_name, sep='\t', index=True, header=None)
    


df_temp = df.copy()

# for i in range(0, len(df.columns), 2):
#     fig, ax = plt.subplots()
#     ax.plot(df.index, df.iloc[:, i], label=df.columns[i])
#     ax.plot(df.index, df.iloc[:, i+1], label=df.columns[i+1])
    
#     ax.set_xlabel('time')
#     ax.set_ylabel('Temp')
#     ax.set_title(f'Plot of {df.columns[i]}')
#     ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
#     plt.xticks(rotation=45)

#     ax.legend()

#     plt.show()

#%%
#analyse
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 14))


ax1.plot(df_prod.index, df_prod['varmepumpe'], label='varmepumpe')
ax1.set_title('varmepumpe')
ax1.set_xlabel('time')
ax1.set_ylabel('watt')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
plt.xticks(rotation=45)


for column in df_rad.columns:
    ax2.plot(df_temp.index, df_rad[column], label=column)
ax2.set_title('radiatorer')
ax2.set_xlabel('time')
ax2.set_ylabel('%')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
plt.xticks(rotation=45)


ax3.plot(df_temp.index, df_temp["radiator_stua_nord"],label='stua_nord')
ax3.plot(df_temp.index, df_temp["radiator_stua_sued"],label='stua_sued')
ax3.plot(df_temp.index, df_temp["radiator_joar"],label='joar')
ax3.set_title('radiatorer temp')
ax3.set_xlabel('time')
ax3.set_ylabel('temp')
ax3.set_ylim(18, 24)
ax3.legend()
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3))  
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
plt.xticks(rotation=45)

ax4.plot(df_temp.index, df_temp['mean_value'], label='utetemp')
ax4.set_title('utetemp')
ax4.set_xlabel('time')
ax4.set_ylabel('temp')
ax4.legend()
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


#%%

start = start_of_month.replace(month = 3, day = 24,hour=14)
slutt = start_of_month.replace(month = 3, day = 24,hour=18)


deg = u'\N{DEGREE SIGN}'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 10))

# Varmepumpe plot
data = df_prod[(df_prod.index >= start) & (df_prod.index <= slutt)]
ax1.plot(data.index, data['varmepumpe'],'-o', label='varmepumpe')
ax1.set_title('Heat Pump Consumption')
ax1.set_ylabel('Watt')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.tick_params(labelbottom=False) 

# Radiatorer plot
data = df_rad[(df_rad.index >= start) & (df_rad.index <= slutt)]
for column in df_rad.columns:
    ax2.plot(data.index, data[column],'-o', label=column)
ax2.set_title('Radiator Output')
ax2.set_ylabel('%')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.tick_params(labelbottom=False) 

# Radiatorer temperature plot
data = df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax3.plot(data.index, data["radiator_stua_nord"],'-o', label='stua_nord')
ax3.plot(data.index, data["radiator_stua_sued"],'-o', label='stua_sued')
ax3.plot(data.index, data["radiator_joar"],'-o', label='joar')
ax3.set_title('Radiator Temperatures')
ax3.set_ylabel(f'Temp [{deg}C]')
ax3.set_ylim(19, 23)
ax3.legend()
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.tick_params(labelbottom=False) 

# Utetemp
ax4.plot(data.index, data['mean_value'],'-o', label='utetemp')
ax4.set_title('Ambient Temperture')
ax4.set_xlabel('Time')
ax4.set_ylabel(f'Temp [{deg}C]')
ax4.legend()
ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()


plt.tight_layout()
plt.show()

#%%

query_bottom = f'SELECT mean("value") AS "bottom" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'esphome_web_d294ae_temp_heating_bottom\' GROUP BY time(1m) FILL(previous)'
query_mid = f'SELECT mean("value") AS "middle" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'esphome_web_d294ae_temp_vv_middle\' GROUP BY time(1m) FILL(previous)'
query_top = f'SELECT mean("value") AS "top" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'esphome_web_d294ae_temp_vv_top\' GROUP BY time(1m) FILL(previous)'

result_bottom = client.query(query_bottom)
result_mid = client.query(query_mid)
result_top = client.query(query_top)


df1 = pd.DataFrame(result_bottom.get_points())
df2 = pd.DataFrame(result_mid.get_points())

df3 = pd.DataFrame(result_top.get_points())


dataframes = [df1, df2, df3]

df = pd.DataFrame()
df["time"] = time_series

for i, df in enumerate(dataframes):
    if df.empty:
       
        zeros_df = pd.DataFrame(0, index=range(len(time_series)),columns=df.columns)
        dataframes[i] = zeros_df

df = pd.DataFrame()
df["time"] = time_series



for i, y in enumerate(dataframes):
        y = y.drop(columns='time') if 'time' in y.columns else y
        y = y.fillna(0)
        df = pd.merge(df, y, how='outer',left_index=True,right_index=True)

df.set_index("time", inplace = True)
df_vp_temp = df.copy()
df_vp_temp1 = df.copy().resample("15T").mean()

for column in df_vp_temp1.columns:
    file_name = f"{column}_data.tsv"
    new_index = ['t' + str(s) for s in range(0, len(df_vp_temp1))]
    df_vp_temp1.index = new_index  
    df_vp_temp1.fillna(method='ffill', inplace=True)
    df_vp_temp1[[column]].to_csv(file_name, sep='\t', index=True, header=None)

#%%
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 12))

ax1.plot(df_prod.index, df_prod['varmepumpe'], label='varmepumpe')
ax1.set_title('varmepumpe')

ax1.set_ylabel('watt')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
plt.xticks(rotation=45)
ax1.legend()

for column in df_vp_temp.columns:
    ax2.plot(df_vp_temp.index, df_vp_temp[column], label=column)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  

ax2.legend()  
ax2.set_ylabel('Deg')
ax2.set_title('Varmetank temp')

ax3.plot(df_temp.index, df_temp['mean_value'], label='utetemp')
ax3.set_title('utetemp')
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
ax3.legend()
ax3.set_ylabel('Deg')
for column in df_rad.columns:
    ax4.plot(df_rad.index, df_rad[column], label=column)
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=3))  
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  

ax4.legend()
ax4.set_title('radiator flow')
ax4.set_ylabel('%')
ax4.set_xlabel('time')
plt.tight_layout()
plt.show()

#%% april 19
fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(5, 1, figsize=(12, 10))
start = start_of_month.replace(day = 19,hour=17, minute= 0)
slutt = start_of_month.replace(day = 19,hour=20, minute = 0)


df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
for column,ax,color in zip(df_vp_temp1.columns,[ax1,ax2,ax3],["orange","green","blue"]):
    ax.plot(df_vp_temp1.index, df_vp_temp1[column], label=column,color=color)
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    #ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15))) 
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.legend()
ax1.set_title('EWH Sensors')
ax2.set_xlabel('Time')
ax1.set_ylabel('Temp (°C)')
ax2.set_ylabel('Temp (°C)')
ax3.set_ylabel('Temp (°C)')
ax2.legend()
ax2.set(xlabel=None)
#for label in ax2.get_xticklabels():
 #   label.set_rotation(45)



df_temp1 = df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax4.plot(df_temp1.index, df_temp1["radiator_stua_nord"],label='stua_nord',color="red")
ax4.set_title('Radiator Temperature Sensor')
# ax4.set_xlabel('Time')
ax4.set_ylabel('Temp (°C)')
ax4.set_ylim(19, 22.2)
ax4.legend()
# ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
# ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
# for label in ax4.get_xticklabels():
#     label.set_rotation(45)

df_temp1 = df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax5.plot(df_temp1.index,df_temp1['mean_value'],label = "$T_{{amb}}$")
ax5.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15))) 
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
for label in ax3.get_xticklabels():
    label.set_rotation(45)
ax5.set_title('Ambient Temperature Sensor')
ax5.legend()
ax5.set_ylabel('Temp (°C)')
ax3.set_xlabel("Time")
ax3.grid()
ax1.grid()
ax2.grid()
ax4.grid()
ax5.grid()
for ax in [ax1,ax2,ax3,ax4]:
    ax.set_xticks([])

for tick in ax5.get_xticks():
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.7)
fig.suptitle("Temperature Data in Pilothouse (Day: 19.April , Hour: 17-20)",fontsize=14)
plt.tight_layout()

plt.show()
#%%
fig, (ax2, ax3,ax4,ax5) = plt.subplots(4, 1, figsize=(12, 10))
start = start_of_month.replace(day = 18,hour=11, minute= 0)
slutt = start_of_month.replace(day = 18,hour=13, minute = 0)


data = df_prod[(df_prod.index >= start) & (df_prod.index <= slutt)]
ax1.plot(data.index, data['varmepumpe'], label='varmepumpe')
ax1.set_title('Heat pump consumption')
ax1.set_xlabel('Time')
ax4.set_ylabel('Temp (°C)')
ax1.legend()


df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
ax2.plot(df_vp_temp1.index, df_vp_temp1["bottom"], label="bottom",color="orange")

ax2.set_title('Temperature of water in Lower section')
ax2.set_xlabel('Time')
ax2.set_ylabel('Temp (°C)')
ax2.legend()

df_temp1 = df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax3.plot(data.index, df_temp1["radiator_stua_nord"],label='stua_nord',color="red")
ax3.set_title('Radiator Temperature Sensor')
ax3.set_xlabel('Time')
ax3.set_ylabel('Temp (°C)')
ax3.set_ylim(20, 22.2)
ax2.set_ylim(24,32)
ax3.legend()
ax3.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10))) 
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
for label in ax3.get_xticklabels():
    label.set_rotation(45)

ax4.plot(data.index,df_temp1['mean_value'],label="utetemp")
ax5.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15))) 
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
    
df_rad1 = df_rad[(df_rad.index >= start) & (df_rad.index <= slutt)]
for column in df_rad1.columns:
    ax5.plot(data.index, df_rad1[column], label=column)
ax5.set_title("Operational Flow of Radiators")
ax5.set_ylim(0,100)
ax5.set_ylabel("% of Max operational flow")
ax5.legend()
ax3.grid()
ax1.grid()
ax2.grid()
ax4.grid()
ax5.grid()

ax3.legend()
ax4.legend()
for ax in [ax1,ax2,ax3,ax4]:
    ax.set_xticks([])
    
for tick in ax5.get_xticks():
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.7)

fig.suptitle("Temperature and Operational Flow of Radioters in Pilothouse (Day: 18 April, Hour: 11-13)",fontsize=14)

plt.tight_layout()


plt.show()
#%%
fig, (ax1,ax2, ax3,ax4,ax5) = plt.subplots(5, 1, figsize=(12, 10))
start = start_of_month.replace(day = 19,hour=17, minute= 0)
slutt = start_of_month.replace(day = 19,hour=20  , minute = 0)


data = df_prod[(df_prod.index >= start) & (df_prod.index <= slutt)]
ax1.plot(data.index, data['varmepumpe'], label='HP')
ax1.set_title('Heat pump consumption')
ax1.set_xlabel('Time')
ax1.set_ylabel('Watt')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
#ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
for label in ax1.get_xticklabels():
    label.set_rotation(45)


df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
for column in df_vp_temp1.columns:
    ax2.plot(df_vp_temp1.index, df_vp_temp1[column], label=column)
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    #ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax2.set_title('EWH Sensors')
ax2.set_xlabel('Time')
ax2.set_ylabel('Temp')
ax2.legend()
for label in ax2.get_xticklabels():
    label.set_rotation(45)

df_temp1 = df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax3.plot(df_temp1.index, df_temp1["radiator_stua_nord"],label='stua_nord')
#ax3.plot(data.index, df_temp1["radiator_stua_sued"],label='stua_sued')
#ax3.plot(data.index, df_temp1["radiator_joar"],label='joar')
ax3.set_title('Radiator Sensors')
ax3.set_xlabel('Time')
ax3.set_ylabel('Temp')
ax2.set_ylim(20, 40)
ax3.set_ylim(20, 25)
ax3.legend()
for label in ax3.get_xticklabels():
    label.set_rotation(45)
for label in ax4.get_xticklabels():
    label.set_rotation(45)
ax4.plot(df_temp1.index,df_temp1['mean_value'],label = "T_amb")
ax4.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))  
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
for label in ax3.get_xticklabels():
    label.set_rotation(45)
    
df_rad1 = df_rad[(df_rad.index >= start) & (df_rad.index <= slutt)]
for column in df_rad1.columns:
    ax5.plot(df_temp1.index, df_rad1[column], label=column)
    ax5.xaxis.set_major_locator(mdates.HourLocator(interval=3))  
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax4.legend()
ax5.legend()
ax5.set_title('radiator flow')
ax5.set_ylabel('%')
ax5.set_xlabel('time')
ax5.legend()
ax3.grid()
ax1.grid()
ax2.grid()
ax4.grid()
plt.tight_layout()


plt.show()
#%%
days = range(1,30)
fig,(ax,ax1,ax2) = plt.subplots(3,1,figsize=(20,14))
fig.suptitle('Temperature sensors in EWH (April 1-14)', fontsize = 16)
for day in range(1,15):
    start = start_of_month.replace(day = day,hour=0, minute= 0)
    slutt = start_of_month.replace(day = day,hour=23, minute = 0)
    df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
    normalized_time = df_vp_temp1.index.map(lambda t: t.replace(year=2023, month=4, day=1))

    for column in df_vp_temp1.columns:
        if column == 'top':
            ax.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax.set_title('Top ')
        elif column == 'middle':
            ax1.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax1.set_title('Midlle')
        else:
            ax2.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax2.set_title('Bottom')
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45) 
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

ax.legend(loc='right')
ax1.legend(loc='right')
ax2.legend(loc='right')
ax.set_ylabel('Deg')
ax1.set_ylabel('Deg')
ax2.set_ylabel('Deg')
plt.tight_layout()
fig,(ax,ax1,ax2) = plt.subplots(3,1,figsize=(20,14))
fig.suptitle('Temperature sensors in EWH (April 15-30)', fontsize = 16)
for day in range(15,31):
    start = start_of_month.replace(day = day,hour=0, minute= 0)
    slutt = start_of_month.replace(day = day,hour=23, minute = 0)
    df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
    normalized_time = df_vp_temp1.index.map(lambda t: t.replace(year=2023, month=1, day=1))

    for column in df_vp_temp1.columns:
        if column == 'top':
            ax.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax.set_title('Top ')
        elif column == 'middle':
            ax1.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax1.set_title('Midlle')
        else:
            ax2.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax2.set_title('Bottom')

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45) 
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

ax.legend(loc='right')
ax1.legend(loc='right')
ax2.legend(loc='right')

ax.set_ylabel('Deg')
ax1.set_ylabel('Deg')
ax2.set_ylabel('Deg')

plt.tight_layout()

#%%
days = range(1,30)
fig,(ax,ax1) = plt.subplots(2,1,figsize=(16,10))
fig.suptitle('Temperature sensors in EWH: Upper section (April 1-14)', fontsize = 16)
for day in range(1,15):
    start = start_of_month.replace(month=4,day = day,hour=0, minute= 0)
    slutt = start_of_month.replace(month=4,day = day,hour=23, minute = 0)
    df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
    normalized_time = df_vp_temp1.index.map(lambda t: t.replace(year=2023, month=4, day=1))

    for column in df_vp_temp1.columns:
        if column == 'top':
            ax.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax.set_title('Top ',fontsize=15)
        elif column == 'middle':
            ax1.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax1.set_title('Middle',fontsize=15)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

s=ax1.get_ylim()
ax.set_ylim(s)

ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45) 

ax.legend(loc='right')
ax1.legend(loc='right')
ax.set_ylabel('Deg',fontsize=14)
ax1.set_ylabel('Deg',fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax1.set_xlabel("Time",fontsize=14)
ax.tick_params(labelsize=12)
ax1.tick_params(labelsize=12)
ax.grid()
ax1.grid()
plt.tight_layout()

fig,(ax,ax1) = plt.subplots(2,1,figsize=(16,10))
fig.suptitle('Temperature sensors in EWH (April 15-30)', fontsize = 16)
for day in range(15,31):
    start = start_of_month.replace(month=4,day = day,hour=0, minute= 0)
    slutt = start_of_month.replace(month=4,day = day,hour=23, minute = 0)
    df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
    normalized_time = df_vp_temp1.index.map(lambda t: t.replace(year=2023, month=4, day=1))

    for column in df_vp_temp1.columns:
        if column == 'top':
            ax.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax.set_title('Top',fontsize=15)
        elif column == 'middle':
            ax1.plot(normalized_time, df_vp_temp1[column], label=f'day {day}')
            ax1.set_title('Midlle',fontsize=15)

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

s=ax1.get_ylim()
ax.set_ylim(s)

plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45) 

ax.legend(loc='right')
ax1.legend(loc='right')
ax.set_ylabel('Deg',fontsize=14)
ax1.set_ylabel('Deg',fontsize=14)
ax.set_xlabel("Time",fontsize=14)
ax1.set_xlabel("Time",fontsize=14)
ax.tick_params(labelsize=12)
ax1.tick_params(labelsize=12)
ax.grid()
ax1.grid()

plt.tight_layout()
#%%
start = start_of_month.replace(day = 19,hour=17, minute= 0)
slutt = start_of_month.replace(day = 19,hour=20  , minute = 0)

df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
lower_min = df_vp_temp1["bottom"].min()
lower_min_time = df_vp_temp1["bottom"].idxmin()
lower_max = df_vp_temp1["bottom"].max()
lower_max_time = df_vp_temp1["bottom"].idxmax()

upper_min = df_vp_temp1["middle"].min()
upper_min_time = df_vp_temp1["middle"].idxmin()
upper_max = df_vp_temp1["middle"].max()
upper_max_time = df_vp_temp1["middle"].idxmax()

time_delta_lower = lower_max_time -lower_min_time
time_delta_upper = upper_max_time -upper_min_time


seconds_lower = -time_delta_lower.total_seconds()
seconds_upper = -time_delta_upper.total_seconds()

hours_lower = seconds_lower / 3600
hours_upper = seconds_upper / 3600
#%%

c = 4186  
rho = 1000  


V_upper = 224 
V_lower = 136  
T_amb = 20 

r = 595 * 10**-3 / 2  
h = 2031 * 10**-3 
A = 2 * np.pi * r * (h + r)  

A_lower = A*(V_lower/(V_lower+V_upper))
A_upper = A*(V_upper/(V_lower+V_upper))

delta_lower = lower_max - lower_min
delta_upper = upper_max - upper_min

m_lower = V_lower
m_upper = V_upper

Q_lower = m_lower * c * delta_lower
Q_upper = m_upper * c * delta_upper

dt_lower = (lower_max + lower_min) / 2 - T_amb
dt_upper = (upper_max + upper_min) / 2 - T_amb

Q_p_lower = Q_lower / seconds_lower
Q_p_upper = Q_upper / seconds_upper

u_lower = Q_p_lower / (A_lower * dt_lower)
u_upper = Q_p_upper / (A_upper * dt_upper)


Q_lower_wh = Q_lower / 3600
Q_upper_wh = Q_upper / 3600

print("Energy loss Lower Section (Wh):", Q_lower_wh)
print("Energy loss Upper Section (Wh):", Q_upper_wh)
print("Thermal Transmittance Lower (W/m²K):", u_lower)
print("Thermal Transmittance Upper (W/m²K):", u_upper)

#%%
#cop
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
start = start_of_month.replace(day = 5,hour=0,minute=0)
slutt = start_of_month.replace(day = 5,hour=20,minute = 0)


data = df_prod[(df_prod.index >= start) & (df_prod.index <= slutt)]
ax1.plot(data.index, data['varmepumpe'], label='varmepumpe')
ax1.set_title('varmepumpe')
ax1.set_xlabel('time')
ax1.set_ylabel('watt')
ax1.legend()
ax1.grid()
ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))  
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
for label in ax1.get_xticklabels():
    label.set_rotation(45)


df_vp_temp1 = df_vp_temp[(df_vp_temp.index >= start) & (df_vp_temp.index <= slutt)]
for column in df_vp_temp1.columns:
    ax2.plot(df_vp_temp1.index, df_vp_temp1[column], label=column)
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30))) 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

ax2.set_title('tank temp')
ax2.set_xlabel('time')
ax2.set_ylabel('temp')
ax2.legend()
ax2.grid()
for label in ax2.get_xticklabels():
    label.set_rotation(45)
    
df_temp3= df_temp[(df_temp.index >= start) & (df_temp.index <= slutt)]
ax3.plot(df_temp3.index, df_temp3['mean_value'], label='utetemp')
ax3.set_title('utetemp')
ax3.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 30))) 
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  
ax3.legend()
ax3.set_ylabel('Deg')

plt.tight_layout()

hp_energy = data["varmepumpe"].sum()/60 
delta_temp = df_vp_temp1["middle"][-1]- df_vp_temp1["middle"][0] 
tank_energy = delta_temp*224* 1.16 

cop =tank_energy/hp_energy
print(cop)

#%%
df_vp_temp.index = pd.to_datetime(df_vp_temp.index)

fig, ax = plt.subplots(figsize=(16, 10)) 

start_date = pd.Timestamp(year=2024, month=4, day=1)  
end_date = pd.Timestamp(year=2024, month=4, day=30) 

df = df_vp_temp.loc[start_date:end_date]
for f in df.columns:
    ax.plot(df.index, df[f], label=f"{f}")


ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.legend() 
ax.set_ylabel("Degrees")
ax.set_xlabel("Days")
ax.set_title("Temperature sensor values of month in 15 min resolution (April-2024)")
ax.grid()
plt.show()  

#%%
query_middle = f'SELECT mean("value") AS "middle" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'esphome_web_d294ae_temp_vv_middle\' GROUP BY time(1m) FILL(previous)'
result_middle = client.query(query_middle)


df1 = pd.DataFrame(result_middle.get_points())
df1.ffill()
df1.fillna(0)

query_top = f'SELECT mean("value") AS "top" FROM "homeassistantdb"."autogen"."°C" WHERE time > \'{dashboardTime}\' AND time < \'{upperDashboardTime}\' AND "entity_id"=\'esphome_web_d294ae_temp_vv_top\' GROUP BY time(1m) FILL(previous)'
result_top = client.query(query_top)


df2 = pd.DataFrame(result_top.get_points())
df2.ffill()
df2.fillna(0)



import numpy as np

startday = 1
endday=8

start = 6000*startday
slutt = 1400*endday


df_energy.reset_index(drop=True, inplace=True)
vp = df_energy['varmepumpe'][start:slutt]
temp= df1["middle"][start:slutt]
temp_t= df2["top"][start:slutt]
temp_amb = df7["mean_value"][start:slutt]
vp.reset_index(drop=True, inplace=True)
temp.reset_index(drop=True, inplace=True)
temp_t.reset_index(drop=True, inplace=True)
temp_amb.reset_index(drop=True, inplace=True)


fig, ( ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(16, 10))


ax1.plot(vp.index, vp)
ax1.set_xlabel('Time steps [min]')
ax1.set_ylabel('Electrical energy [Wh]')
ax1.set_title('Heat pump energy consumption over a day')
ax1.grid(True)

ax2.plot(temp.index, temp, label = "middle")
ax2.plot(temp.index,temp_t, label = "top")
ax2.set_xlabel('Time steps [min]')
ax2.set_ylabel('Degrees')
ax2.set_title('Temperature increase in "middle" and "top" water')
ax2.grid(True)

ax2.legend()

ax3.plot(temp_amb.index,temp_amb, label = "amb")
ax3.set_xlabel('Time steps [min]')
ax3.set_ylabel('Degrees')
ax3.set_title('Ambient Temperature')
ax3.grid(True)

ax3.legend()


start_1 = 160 *startday
slutt_1 = 200*startday

ax1.axvline(x=start_1, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=start_1, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=start_1, color='r', linestyle='--', linewidth=2)

ax1.axvline(x=slutt_1, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=slutt_1, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=slutt_1, color='r', linestyle='--', linewidth=2)

start_2 = 1700*startday
slutt_2 = 1750*startday

ax1.axvline(x=start_2, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=start_2, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=start_2, color='r', linestyle='--', linewidth=2)


ax1.axvline(x=slutt_2, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=slutt_2, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=slutt_2, color='r', linestyle='--', linewidth=2)

start_3 = 3040*startday
slutt_3 = 3085*startday

ax1.axvline(x=start_3, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=start_3, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=start_3, color='r', linestyle='--', linewidth=2)


ax1.axvline(x=slutt_3, color='r', linestyle='--', linewidth=2)
ax2.axvline(x=slutt_3, color='r', linestyle='--', linewidth=2)
ax3.axvline(x=slutt_3, color='r', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()

#%%
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(16, 10))

start_1_del = start_1 + 10
start_2_del = start_2 + 10
start_3_del = start_3 + 10 

ax1.plot(vp[start_1:slutt_1].index, vp[start_1:slutt_1], label ='HP_1')
ax1.plot(vp[start_2:slutt_2].index, vp[start_2:slutt_2], label ='HP_2')
ax1.plot(vp[start_3:slutt_3].index, vp[start_3:slutt_3], label ='HP_3')

ax2.plot(temp[start_1_del:slutt_1].index, temp[start_1_del:slutt_1], label = "middle_1")
ax2.plot(temp[start_2_del:slutt_2].index, temp[start_2_del:slutt_2], label = "middle_2")
ax2.plot(temp[start_3_del:slutt_3].index, temp[start_3_del:slutt_3], label = "middle_3")
ax2.plot(temp_t[start_1_del:slutt_1].index, temp_t[start_1_del:slutt_1], label = "top_1")
ax2.plot(temp_t[start_2_del:slutt_2].index, temp_t[start_2_del:slutt_2], label = "top_2")
ax2.plot(temp_t[start_3_del:slutt_3].index, temp_t[start_3_del:slutt_3], label = "top_3")

ax3.plot(temp_amb[start_1:slutt_1].index, temp_amb[start_1:slutt_1], label ="ambient_temp_1")
ax3.plot(temp_amb[start_2:slutt_2].index, temp_amb[start_2:slutt_2],  label ="ambient_temp_2")
ax3.plot(temp_amb[start_3:slutt_3].index, temp_amb[start_3:slutt_3],  label ="ambient_temp_3")

ax1.set_xlabel('Time steps [min]')
ax1.set_ylabel('Electrical energy [Wh]')
ax1.set_title('Heat pump energy consumption during heating prcoess ')
ax1.grid(True)

ax2.set_xlabel('Time steps [min]')
ax2.set_ylabel('Degrees')
ax2.set_title('Temperature increase in "middle" and "top" water during heating (5 min delay)')
ax2.grid(True)


ax3.set_xlabel('Time steps [min]')
ax3.set_ylabel('Degrees')
ax3.set_title('Ambient Temperature')
ax3.grid(True)
plt.tight_layout()



#%%
temp_diff_1 = temp.iloc[slutt_1]-temp.iloc[start_1_del]
temp_diff_2 = temp.iloc[slutt_2]-temp.iloc[start_2_del]
temp_diff_3 = temp.iloc[slutt_3]-temp.iloc[start_3_del]

e = 0.9

energy_middle_1 = temp_diff_1 * (4184 *220 / 3600)*(slutt_1-start_1_del)/60
W_1 = vp[start_1:slutt_1].sum()
Q_1 = energy_middle_1
t_1 = temp_amb[start_1:slutt_1].mean()

cop_1 = Q_1/(W_1*e)

energy_middle_2 = temp_diff_2  * (4184 * 220 / 3600)*(slutt_2-start_2_del)/60
W_2 = vp[start_2:slutt_2].sum()
Q_2 = energy_middle_2
t_2 = temp_amb[start_2:slutt_2].mean()

cop_2 = Q_2/(W_2*e)

energy_middle_3 = temp_diff_3  * (4184 * 220 / 3600)*(slutt_3-start_3_del)/60
W_3 = vp[start_3:slutt_3].sum()
Q_3 = energy_middle_3
t_3 = temp_amb[start_3:slutt_3].mean()

cop_3 = Q_3/(W_3*e)



from sklearn.linear_model import LinearRegression

data = {
    "Temperature": [t_1, t_2, t_3],
    "COP": [cop_1, cop_2, cop_3]
}

df = pd.DataFrame(data)

model = LinearRegression()

X = np.array(df['Temperature']).reshape(-1, 1)
y = df['COP']
model.fit(X, y)

slope = model.coef_
intercept = model.intercept_

df['Predicted_COP'] = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(df['Temperature'], df['COP'], color='blue', label='Actual COP')
plt.plot(df['Temperature'], df['Predicted_COP'], color='red', label='Regression Line')
plt.title('COP vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('COP')
plt.legend()
plt.grid(True)
plt.show()

#%%
f =df_prod1["fixed_load"].resample("H").mean().sum()
f1 = (df_prod1['solar_prod']-df_prod1['export'])

fig,(ax,ax1) = plt.subplots(2,1,figsize = (14,10))
ax.plot(f1.index,df_prod1['solar_prod']/1000,label="Solar production")
ax.plot(f1.index,df_prod1['export']/1000,label="Export")
ax1.plot(f1.index,f1/1000,label="Solar consumption")
ax.legend(fontsize=12)
ax.grid()
ax.set_ylabel("kW",fontsize=14)
ax.set_xlabel("Day",fontsize=14)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d')) 
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45,fontsize=12)
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=14)
ax.set_title("Solar production against Export (April)")
plt.tight_layout()
ax1.legend(fontsize=12)
ax1.grid()
ax1.set_ylabel("kW",fontsize=14)
ax1.set_xlabel("Day",fontsize=14)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45,fontsize=12)
plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=14)
plt.tight_layout()
plt.show()

i = (df_prod3["import"].sum()+df_prod3["solar_prod"].sum()-df_prod3["export"].sum())/1000
hp = df_prod3["varmepumpe"].sum()

labels = 'HP', 'Fixed load'
sizes = [hp/1000, f/1000]
colors = ['gold', 'lightskyblue']
explode = (0.1, 0) 

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.1f}%\n({v:d} kWh)'.format(p=pct, v=val)
    return my_autopct


plt.figure(figsize=(8, 6)) 
plt.pie(sizes, explode=None, labels=None, colors=colors,
        autopct=make_autopct(sizes), shadow=True, startangle=140)
plt.axis('equal') 
plt.title('Total Energy Consumption of Pilothouse April (2024)')


plt.legend(labels, title="Components", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.tight_layout()
plt.show()

#%%
import matplotlib.ticker as ticker

df_p = pd.read_csv("Masteroppgave/data/Day-ahead Prices_202401010000-202501010000.csv")
df_p['Timestamp'] = pd.to_datetime(df_p['MTU (CET/CEST)'].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')
df_p.set_index('Timestamp', inplace=True)


df_prices = df_p[['Day-ahead Price [EUR/MWh]']]

euro_to_nok =11.5
Mwh_to_wh =10**(-6)
df_p_m = df_prices[df_prices.index.month == 4]
df_p_m_k = df_p_m.astype(float)*( euro_to_nok*Mwh_to_wh)*10**3
price = df_p_m_k.resample("15T").ffill()

if price.index.max() < f1.index.max():

    new_index = pd.date_range(start=price.index[-1] + pd.Timedelta(minutes=15), end=f1.index.max(), freq='15T')
    price = price.reindex(price.index.union(new_index)).ffill()

fig,(ax,ax1,ax2) = plt.subplots(3,1,figsize = (14,10),sharex=True)
ax.plot(f1.index,df_prod1['solar_prod']/1000,color = 'red')
ax1.plot(f1.index,df_temp['mean_value'].resample("15T").mean(), color = 'blue')
ax2.plot(f1.index,price,color = 'orange')
ax.legend(fontsize=12)
ax.grid()
ax.set_ylabel("kW",fontsize=14)
ax.set_xlabel("Day",fontsize=14)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d')) 
plt.setp(ax.xaxis.get_majorticklabels(), fontsize=12)
plt.setp(ax.yaxis.get_majorticklabels(), fontsize=14)
ax.set_title("Solar production (April)")
plt.tight_layout()
ax1.legend(fontsize=12)
ax1.grid()
ax1.set_ylabel(f"Temperature [{deg} C]",fontsize=14)
ax1.set_xlabel("Day",fontsize=14)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1)) 
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=12)
plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=14)
ax1.set_title("Ambient Temperature")
plt.tight_layout()
ax2.legend(fontsize=12)
ax2.grid()
ax2.set_ylabel("Nok/kWh",fontsize=14,)
ax2.set_xlabel("Day",fontsize=14)
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))  
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d'))  
plt.setp(ax2.xaxis.get_majorticklabels(),fontsize=12)
plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=14)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}'))
ax2.set_title("Spot price ")
plt.tight_layout()

dates_to_mark = ['2024-04-05','2024-04-06' ,'2024-04-18','2024-04-19','2024-04-20', '2024-04-21'] 
for date in dates_to_mark:
    for axx in [ax, ax1, ax2]:
        axx.axvline(pd.to_datetime(date), color='black', linestyle='--', linewidth=2) 

plt.show()
#%%
fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=(14, 10))

daily_solar_prod = df_prod1['solar_prod'].resample('D').mean()/1000  
daily_temp_mean = df_temp['mean_value'].resample('D').mean() 
daily_fixed_load = df_prod1['fixed_load'].resample('D').mean()/1000  

offset = pd.Timedelta(hours=12) 


ax.bar(daily_solar_prod.index + offset, daily_solar_prod.values, color='red', width=0.8, label="Solar production")
ax1.bar(daily_temp_mean.index + offset, daily_temp_mean.values, color='blue', width=0.8, label="Average Temperature")
ax2.bar(daily_fixed_load.index + offset, daily_fixed_load.values, color='orange', width=0.8, label="Fixed Load")


for axx in [ax, ax1, ax2]:
    axx.legend(fontsize=12)
    axx.grid()
    axx.set_xlabel("Day", fontsize=14)
    axx.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axx.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
    plt.setp(axx.xaxis.get_majorticklabels(), fontsize=12)
    plt.setp(axx.yaxis.get_majorticklabels(), fontsize=14)

ax.set_ylabel("kWh", fontsize=14)
ax1.set_ylabel("Temperature (°C)", fontsize=14)
ax2.set_ylabel("kWh", fontsize=14)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}'))  # Ensure integer y-ticks on ax2

ax.set_title("Daily Solar Production")
ax1.set_title("Daily Average Temperature")
ax2.set_title("Daily Fixed Load")
plt.tight_layout()
plt.show()
