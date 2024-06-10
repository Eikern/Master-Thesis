import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3



data_folder = "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data"
folder = 'sensitivity'
days = [20, 18, 5]

#%%
def read_data(table_name, con):
    query = f"SELECT * FROM {table_name}"
    try:
        df = pd.read_sql_query(query, con)
        if df.empty:
            raise ValueError("No data returned from the database.")
        return df
    except Exception as e:
        print(f"Error querying table {table_name}: {e}")
        return pd.DataFrame()
    
df_i = pd.DataFrame(index = pd.date_range(start='2024-01-01', periods=1440, freq='T'))

data = {day: pd.DataFrame() for day in days}
solar={day: pd.DataFrame() for day in days}
hp_lower = {day: pd.DataFrame() for day in days}
hp_upper = {day: pd.DataFrame() for day in days}

for day in days:
    db_path = f"C:/Users/eirik/Documents/GAMS/Studio/workspace/sensitivity/results_master_{day}_5_min_import_power_sens.db"
    with sqlite3.connect(db_path) as con:
        column_name = f"{day}"
        data[day] = read_data('results', con)
        solar[day] = read_data('solar_values', con)
        hp_lower[day] = read_data('hp_lower_values', con)
        hp_upper[day] = read_data('hp_upper_values', con)
        
    liste = ["solar_prod","fixed_load","mean_value"]
    
    for l in liste:
            x= pd.read_csv(f"C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/5_min_data/{l}_5_min_data_{day}_gams.tsv", sep='\t',names=["time", "value"])
            r = pd.date_range(start='2024-01-01', periods=len(x), freq='5T')
            x.set_index(r, inplace=True)
            x_reindexed = x.reindex(df_i.index, method='ffill')
            df_i[f"{l}_5_min_{day}"] = x_reindexed["value"]
            if l == "mean_value":
                df_i[f"{l}_5_min_{day}"]  = df_i[f"{l}_5_min_{day}"] 
            else:
                df_i[f"{l}_5_min_{day}"] =df_i[f"{l}_5_min_{day}"]/5
               
#%%
def process_dataframe(df):
    df['t'] = df['t'].str.extract('(\d+)').astype(int)
    full_range = pd.DataFrame({
        't': np.arange(1, 289),
    })
    
    unique_iz = df[['i', 'z']].drop_duplicates()
    
    reindexed_dfs = []
    
    for _, row in unique_iz.iterrows():
        i, z = row['i'], row['z']
        df_iz = df[(df['i'] == i) & (df['z'] == z)]
        df_iz_full = pd.merge(full_range, df_iz, on='t', how='left')
        df_iz_full['i'] = df_iz_full['i'].fillna(i)
        df_iz_full['z'] = df_iz_full['z'].fillna(z)
        df_iz_full['value'] = df_iz_full['value'].fillna(0)
        reindexed_dfs.append(df_iz_full)
    
    df_full = pd.concat(reindexed_dfs)
    df_full = df_full.sort_values(by=['t', 'i', 'z']).reset_index(drop=True)
    df_full['t'] = 't' + df_full['t'].astype(str)
    
    return df_full

for key in solar.keys():
    solar[key] = process_dataframe(solar[key])
    hp_lower[key] = process_dataframe(hp_lower[key])
    hp_upper[key] = process_dataframe(hp_upper[key])


def assign_datetime_index(df):
    unique_iz = df[['i', 'z']].drop_duplicates()
    
    datetime_dfs = []
    for _, row in unique_iz.iterrows():
        i, z = row['i'], row['z']
        df_iz = df[(df['i'] == i) & (df['z'] == z)].copy()
        
        date_range = pd.date_range(start='2024-01-01', periods=288, freq='5T')
        df_iz['datetime'] = date_range
        
        datetime_dfs.append(df_iz)
    
    df_with_datetime = pd.concat(datetime_dfs)
    
    return df_with_datetime

for key,key,key in zip(solar.keys(),hp_lower.keys(),hp_upper.keys()):
    solar[key] = assign_datetime_index(solar[key])
    hp_lower[key] = assign_datetime_index(hp_lower[key])
    hp_upper[key] = assign_datetime_index(hp_upper[key])
    

def sum_hp_values_above_zero(solar, hp_lower, hp_upper, solar_prod, day):
    results = []
    df_day = solar[day]
    
    solar_prod_resampled = solar_prod.resample('5T').sum()
    
    solar_prod_positive = solar_prod_resampled > 0
    positive_intervals = solar_prod_positive.index[solar_prod_positive].tolist()
    
    for (i, z), group in df_day.groupby(['i', 'z']):
        hp_lower_filtered = hp_lower[day][(hp_lower[day]['i'] == i) & (hp_lower[day]['z'] == z)]
        hp_lower_filtered_solar = hp_lower[day][(hp_lower[day]['i'] == i) & (hp_lower[day]['z'] == z) & (hp_lower[day]['datetime'].isin(positive_intervals))]
        hp_lower_sum = hp_lower_filtered['value'].sum()
        hp_lower_sum_solar = hp_lower_filtered_solar['value'].sum()
        
        hp_upper_filtered = hp_upper[day][(hp_upper[day]['i'] == i) & (hp_upper[day]['z'] == z)]
        hp_upper_filtered_solar = hp_upper[day][(hp_upper[day]['i'] == i) & (hp_upper[day]['z'] == z) & (hp_upper[day]['datetime'].isin(positive_intervals))]
        
        
        hp_upper_sum = hp_upper_filtered['value'].sum()
        hp_upper_sum_solar = hp_upper_filtered_solar['value'].sum()
        
        hp_lower_ratio= hp_lower_sum_solar/hp_lower_sum
        hp_upper_ratio = hp_upper_sum_solar/hp_upper_sum
        
        results.append({
            'i': i,
            'z': z,
            f'hp_lower_{day}': hp_lower_sum,
            f'hp_upper_{day}': hp_upper_sum,
            f'hp_lower_solar_{day}': hp_lower_sum_solar,
            f'hp_upper_solar_{day}': hp_upper_sum_solar,
            f'hp_lower_ratio_{day}': hp_lower_ratio,
            f'hp_upper_ratio_{day}': hp_upper_ratio,
        })
    
    results_df = pd.DataFrame(results)
    
    return results_df

data1 = {day:pd.DataFrame() for day in days}

for key in solar.keys():
    day = key
    solar_prod = df_i[f'solar_prod_5_min_{day}']
    hp_values_sum = sum_hp_values_above_zero(solar, hp_lower, hp_upper, solar_prod, day)
    data1[day] = hp_values_sum
#%%
for day in days:
    pivot_df1 = data1[day].pivot(index='z', columns='i', values=f'hp_lower_ratio_{day}')
    pivot_df2 = data1[day].pivot(index='z', columns='i', values=f'hp_upper_ratio_{day}')
    
    fig,ax = plt.subplots(figsize=(16,10))
    #ax = sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".1f", linewidths=.5)
    ax = sns.heatmap(pivot_df1, annot=True, annot_kws={'size': 12},fmt=".2f", ax=ax,linewidths=.8)
    ax.set_title(f'Heatmap of the ratio between $HP_l$ consumption during solar production and the whole day for $(DHW * z)$ and ($L_t$ *i) Day {day}',fontsize = 14)
    ax.set_xlabel('i',fontsize = 14)
    ax.set_ylabel('$z$',fontsize = 14)
    plt.xticks(rotation=0,fontsize = 12) 
    plt.yticks(rotation=0,fontsize = 12)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(f"heatmap_loadshift_{day}_lower.png")

    fig,ax = plt.subplots(figsize=(16,10))
    #ax = sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".1f", linewidths=.5)
    ax = sns.heatmap(pivot_df2, annot=True, annot_kws={'size': 12},fmt=".2f", ax=ax,linewidths=.8)
    ax.set_title(f'Heatmap of the ratio between $HP_u$ consumption during solar production and the whole day for $(DHW * z)$ and ($L_t$ *i) Day {day}',fontsize = 14)
    ax.set_xlabel('L*i',fontsize = 14)
    ax.set_ylabel('$DHW*z$',fontsize = 14)
    plt.xticks(rotation=0,fontsize = 12)  
    plt.yticks(rotation=0,fontsize = 12)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(f"heatmap_loadshift_{day}_hp_upper.png")

#%%
print(hp_lower)
z_values = ['z1', 'z2', 'z3', 'z4', 'z5']
i_values = ['i1', 'i2', 'i3', 'i4', 'i5']

def resample_to_hourly(df):
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        resampled_df = df.resample('H').mean().reset_index()
        return resampled_df
    return df

for day in days:
    df = hp_lower[day]
    df = resample_to_hourly(df)
    fig, axs = plt.subplots(nrows=len(z_values), ncols=len(i_values), figsize=(20, 15), sharex=True, sharey=True)
    fig.suptitle(f'Hourly Behavior for Each Scenario and Size on {day}')

    for z_idx, z in enumerate(z_values):
        for i_idx, i in enumerate(i_values):
            ax = axs[z_idx, i_idx]
            subset = df[(df['z'] == z) & (df['i'] == i)]
            if not subset.empty:
                ax.plot(subset['datetime'], subset['value'])
            ax.set_title(f'Scenario {z} Size {i}')
    
    for ax in axs.flat:
        ax.label_outer()

    plt.xlabel('Time (hours)')
    plt.ylabel('Values')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
#%%

for day in days:
    solar_prod = df_i[f"solar_prod_5_min_{day}"].resample("5T").sum()
    fixed_load = df_i[f"fixed_load_5_min_{day}"].resample("5T").sum()
    objective_values = data[day][data[day]['results'] == 'Costs'][['i', 'z', 'value']].rename(columns={'value': 'objective_value'})
    i_values = data[day][data[day]['results'] == 'i'][['i', 'z', 'value']].rename(columns={'value': 'factor_dhw'})
    z_values = data[day][data[day]['results'] == 'z'][['i', 'z', 'value']].rename(columns={'value': 'factor_load'})


    merged_df = pd.merge(objective_values, z_values, on=['i', 'z'])
    merged_df = pd.merge(merged_df, i_values, on=['i', 'z'])
    

    aggregated_df = merged_df.groupby(['factor_load', 'factor_dhw']).agg({'objective_value': 'mean'}).reset_index()
    pivot_df = aggregated_df.pivot(index='factor_load', columns='factor_dhw', values='objective_value')
        
    fig,ax = plt.subplots(figsize=(16,10))
    #ax = sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".1f", linewidths=.5)
    ax = sns.heatmap(pivot_df, annot=True, annot_kws={'size': 12},fmt=".2f", ax=ax,linewidths=.8)
    ax.set_title(f'Day {day}: Heatmap of Costs for fixed loads ($L*i$) and DHW magnitudes ($DHW_t$ * $z$)',fontsize = 14)
    ax.set_xlabel('i',fontsize = 14)
    ax.set_ylabel('$z$',fontsize = 14)
    plt.xticks(rotation=0,fontsize = 12)
    plt.yticks(rotation=0,fontsize = 12)
    plt.tight_layout()
    
    plt.show()
    plt.savefig(f"heatmap_{day}.png")

