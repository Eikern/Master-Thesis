import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


start_time = pd.Timestamp("2024-03-01 00:00:00")
end_time = pd.Timestamp("2024-04-01 00:00:00")

time_range_min = pd.date_range(start=start_time, end=end_time, freq='T')

time_min = 60

shower_energy = 6000/15 
fauset_15_energy = 0 
fauset_10_energy = 5600/10 
fauset_5_energy = 2800/5 






consumption_schedule = {
    'shower': [((7, 0), (7, 15)), ((7, 30), (7, 45)),((18, 0), (18, 15)),((21, 0), (21, 15)), ((22, 0), (22, 15))],  # Start and end times for showers
    'fauset_15':[((20, 0), (20, 15))],
    'fauset_10':[((19, 0), (19, 10))],
    'fauset_5':[((17, 0), (17,5))]
}

def generate_consumption_events(schedule, time_range, shower_energy, fauset_15_energy,fauset_10_energy,fauset_5_energy):
    df = pd.DataFrame(index=time_range, columns=schedule.keys(), data=0)
    for day in pd.date_range(start=time_range[0].date(), end=time_range[-1].date(), freq='D'):
        for appliance, timings in schedule.items():
            for start, end in timings:
                start_hour, start_minute = start
                end_hour, end_minute = end
                start_time = pd.Timestamp(day) + pd.Timedelta(hours=start_hour, minutes=start_minute)
                end_time = pd.Timestamp(day) + pd.Timedelta(hours=end_hour, minutes=end_minute)
                end_time += pd.Timedelta(minutes=-1)  
                if appliance == 'fauset_15':
                    energy_value = fauset_15_energy
                elif appliance == 'fauset_10':
                    energy_value = fauset_10_energy
                elif appliance == 'fauset_5':
                    energy_value = fauset_5_energy
                else:
                    energy_value = shower_energy
                df.loc[start_time:end_time, appliance] += energy_value
    return df



path_hour = 'hour_data'
path_30_min = "30_min_data"
path_15_min = "15_min_data"
path_5_min = "5_min_data"
path_min = "1_min_data"

paths = [path_hour,path_30_min,path_15_min,path_5_min,path_min]
dfss = []

for path in paths:
    os.makedirs(path, exist_ok=True)
    
times =['H','30T','15T','5T','T']
for path,time,s in zip(paths,times,[1,2,4,12,60]):
    time_range = time_range_min[:-1]
    c= generate_consumption_events(consumption_schedule, time_range_min, shower_energy, fauset_15_energy,fauset_10_energy,fauset_5_energy)
    df= c.resample(time).sum()
    df["fauset"]=df["fauset_15"]+df["fauset_10"]+df["fauset_5"]
    df.drop(columns=["fauset_15", "fauset_10", "fauset_5"], inplace=True)
    df = df[:-1]
    dfss.append(df)
    print(len(df))
    plot_file_path = os.path.join(path, f'step_plot_{path}.png')
    #plot_step_data(df, f'Step Plot for {path}', plot_file_path)
    filtered_df = df[df.index.day == 1]
    fig, ax = plt.subplots(figsize=(16,10))
    for column in filtered_df.columns:
        ax.plot(filtered_df.index, filtered_df[column], label=column)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('DHW simulated Demand '+str(60/s) +" min")
    ax.grid()
    ax.legend()
    ax.set_ylabel("Wh")
    ax.set_xlabel("Time")
    
    plt.setp(ax.get_xticklabels(), rotation=45)
    df.reset_index(drop=True, inplace=True)
    file_name = f'event_{path}.tsv'
    file_path = os.path.join(path, file_name)
    df.to_csv(file_path, sep='\t')
    #Gams
    for c in df.columns:
        df['time'] = ['t' + str(i) for i in range(1, len(df) + 1)]  
        df.set_index('time', inplace=True)
        file_name = f'{c}_{path}_gams.tsv'
        file_path = os.path.join(path, file_name)
        df.iloc[0:(24*s)][c].to_csv(file_path,sep="\t",index=True,header = None)
