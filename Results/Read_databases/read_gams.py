import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter



data_folder = "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data"
scenarios = ["s1",'s2','s3']
folders_scenarios =['scenario1','scenario2','scenario3']
folders = ['basecase_20', 'basecase_18', 'basecase_5']
days = [20, 18, 5] 
tid = [5,1]
resolutions = ['5_min', '1_min']
objectives = ["total_cost","import_power"]

table_names = [
    "hp_upper", "hp_lower", "temp_upper", "temp_lower", "temp_room",
    "import_power", "solar", "export_power", "hp_lower_on", "hp_upper_on",
    "heat_in_lower", "lower_loss", "upper_loss", "heat_in_upper",
    "heat_in_room", "room_loss", "heat_lower", "heat_upper", "heat_room",
    "total_cost", "import_energy", "export_energy"
]

cost_sale_days = ["import_cost","export_sale"]

cost_sale_day = {day:pd.DataFrame() for day in days}
dfs_resolutions = {obj:{s:{res: pd.DataFrame() for res in resolutions} for s in scenarios}for obj in objectives}
dfs_reduction = {obj:{s:{res: pd.DataFrame() for res in resolutions} for s in scenarios}for obj in objectives}

def format_yticks(y, pos):
    return f'{int(y)}'

def read_data(table_name, con):
    if table_name in cost_sale_days:
        query = f"SELECT value FROM {table_name}"
    else:
        query = f"SELECT level FROM {table_name}" if table_name != "total_cost" else "SELECT level FROM scalarvariables"
    try:
        df = pd.read_sql_query(query, con)
        if df.empty:
            raise ValueError("No data returned from the database.")
        return df
    except Exception as e:
        print(f"Error querying table {table_name}: {e}")
        return pd.DataFrame()
    
for s,sf in zip(scenarios,folders_scenarios):
    for folder, day in zip(folders, days):
        for resolution in resolutions:
            for obj in objectives:
                x = folder[-2:]
                y = s[-1]
                z = sf[-1]
                if x == '_5':
                    x = 5
                if (y== z) and (int(x) == day):
                    db_path = f"C:/Users/eirik/Documents/GAMS/Studio/workspace/{sf}/{folder}/results_master_{day}_{resolution}_{s}_{obj}.db"
                    with sqlite3.connect(db_path) as con:
                        for table_name in table_names:
                            df = read_data(table_name, con)
                            df_red = read_data(table_name, con)
                            if not df.empty:
                                column_name = f"{table_name}_{day}"
                                if column_name not in dfs_resolutions[obj][s][resolution]:
                                    dfs_resolutions[obj][s][resolution][column_name] = df.squeeze() 
                                    dfs_reduction[obj][s][resolution][column_name] = df_red.squeeze()
                        
for day,folder in zip(days,folders):
    db_path = f"C:/Users/eirik/Documents/GAMS/Studio/workspace/scenario3/{folder}/results_master_{day}_5_min_s3_import_power.db"
    with sqlite3.connect(db_path) as con:
        for n in cost_sale_days:   
            df = read_data(n, con)
            if not df.empty:
                column_name = f"{n}_{day}"
                if column_name not in  cost_sale_day[day]:
                    df.index=pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                    cost_sale_day[day][column_name] = df.squeeze()
                    
#%%
df_i = pd.DataFrame(index = pd.date_range(start='2024-01-01', periods=1440, freq='T'))
for folder, day in zip(folders, days):
    for resolution,t in zip(resolutions,tid):
        liste = ["solar_prod","fixed_load","mean_value"]
        for l in liste:
             x= pd.read_csv(f"{data_folder}/{resolution}_data/{l}_{resolution}_data_{day}_gams.tsv", sep='\t',names=["time", "value"])
             r = pd.date_range(start='2024-01-01', periods=len(x), freq=f'{t}T')
             x.set_index(r, inplace=True)
             x_reindexed = x.reindex(df_i.index, method='ffill')
             df_i[f"{l}_{resolution}_{day}"] = x_reindexed["value"]
             if l == "mean_value":
                 df_i[f"{l}_{resolution}_{day}"]  = df_i[f"{l}_{resolution}_{day}"] 
             else:
                 df_i[f"{l}_{resolution}_{day}"] =df_i[f"{l}_{resolution}_{day}"]/t
                
             
df_i.sort_index(axis=1)



#%%

price_18 = pd.read_csv(f"{data_folder}/prices_april_18_gams.tsv", sep='\t', header=None, names=["time", "value"])
price_20 =pd.read_csv(f"{data_folder}/prices_april_20_gams.tsv", sep='\t', header=None, names=["time", "value"])
price_5 =pd.read_csv(f"{data_folder}/prices_april_5_gams.tsv", sep='\t', header=None, names=["time", "value"])

prices = [price_20["value"]*10**3,price_18["value"]*10**3,price_5["value"]*10**3]
for p in prices:
    p.index = pd.date_range(start='2024-01-01', periods=len(p), freq='H')

energy_types = ['import_energy', 'export_energy']

energy_df = pd.DataFrame()
costs = pd.DataFrame()
sc = {}
ss={}
for obj in objectives:
    for s in scenarios:
        for resolution, t in zip(resolutions, tid):
            if resolution in dfs_resolutions[obj][s]:
                for day in days:
                    columns_to_extract = [f"{energy_type}_{day}" for energy_type in energy_types if f"{energy_type}_{day}" in dfs_resolutions[obj][s][resolution].columns]
                    column_name = f"total_cost_{day}"
                    
                    if column_name in dfs_resolutions[obj][s][resolution].columns:
                        cost_data = dfs_resolutions[obj][s][resolution][[column_name]]
                        cost_data.columns = [f"{obj}_{s}_{day}_{resolution}"]
                        
                        costs = pd.concat([costs, cost_data], axis=1)
                        
                        dfs_resolutions[obj][s][resolution].drop(columns=[column_name], inplace=True)
        
                    if columns_to_extract:
                        selected_columns = dfs_resolutions[obj][s][resolution][columns_to_extract].iloc[0:24]
                        selected_columns.rename(columns={col: f"{col}_{resolution}_{s}_{obj}" for col in columns_to_extract}, inplace=True)
                        energy_df = pd.concat([energy_df, selected_columns], axis=1)
                        dfs_resolutions[obj][s][resolution].drop(columns=columns_to_extract, inplace=True)
                
                dfs_resolutions[obj][s][resolution].index = pd.date_range(start='2024-01-01', periods=len(dfs_resolutions[obj][s][resolution]), freq=f'{t}T')
    
    energy_df.index = pd.date_range(start='2024-01-01', periods=len(energy_df), freq='H')
    
    
for obj in objectives:
    for resolution in resolutions:
        for day in days:
            for s in scenarios:
                try:
                    solar_sum = dfs_resolutions[obj][s][resolution][f"solar_{day}"].sum()
                    hp_l = dfs_resolutions[obj][s][resolution][f"hp_lower_{day}"]
                    hp_u = dfs_resolutions[obj][s][resolution][f"hp_upper_{day}"]
                    prod_sum = df_i[f"solar_prod_{resolution}_{day}"].sum()
                    f = df_i[f"fixed_load_{resolution}_{day}"].sum()
                    hp_sum = (hp_l+hp_u).sum()
                    if s == "s1":
                        dem_sum=hp_sum
                    else:
                        dem_sum=f+hp_sum
                        
                    ss_value = solar_sum / dem_sum
                    sc_value = solar_sum / prod_sum
                    ss[f'{obj}_{s}_{day}_{resolution}']= float(ss_value)*100
                    sc[f'{obj}_{s}_{day}_{resolution}']= float(sc_value)*100
                except KeyError as e:
                    print(f"Missing data for scenario: {s}, resolution: {resolution}, day: {day}. Error: {e}")
                except Exception as e:
                    print(f"An error occurred for scenario: {s}, resolution: {resolution}, day: {day}. Error: {e}")
                    
sc_df = pd.DataFrame(sc, index=[0]).T
ss_df = pd.DataFrame(ss, index=[0]).T

dfs = [sc_df,ss_df]
name = ["Self Consumption", "Self Sufficiency"]
obj_name = ["Costs", "Import"]
hatch_patterns = {
    'Import': '--',
    'Costs': '++'
}
#%% Self consumption
import os
filepath = "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/Figures_overleaf/results/SS_SC"
#%%
# c= costs.iloc[0]
# print(c)
# sorted_costs = c.sort_index()

# # Reset the index and create a DataFrame
# sorted_costs_df = sorted_costs.reset_index()
# sorted_costs_df.columns = ['Index', 'Cost']
# sorted_costs_df["objective"] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[0])
# sorted_costs_df['Scenario'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[2])
# sorted_costs_df['Day'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[3])
# sorted_costs_df['Resolution'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[4])
# sorted_costs_df['Day_Resolution'] = sorted_costs_df.apply(lambda x: f"Day {x['Day']}:\n {x['Resolution']} min", axis=1)

# objectives1 = sorted_costs_df['objective'].unique()

# for obj in objectives1:
#     # Plotting using Seaborn
#     plt.figure(figsize=(16, 10))
#     ax = sns.barplot(data=(sorted_costs_df[sorted_costs_df['objective'] == obj]), x='Day_Resolution', y='Cost', hue='Scenario', ci=None, palette='tab10', dodge=True)
    
#     plt.title(f'Objective: {obj} Cost comparison across days and resolutions for different scenarios', fontsize=16)  # Adjust title fontsize
#     plt.ylabel('Costs [NOK]', fontsize=14)  # Adjust y-axis label fontsize
#     plt.xlabel('Day and Resolution', fontsize=14)  # Adjust x-axis label fontsize
#     plt.xticks( fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.legend(title='Resolution', title_fontsize='13', fontsize='12', loc='upper left')
#     ax.yaxis.grid(True)  # Enable the grid
#     ax.set_axisbelow(True)  # Ensures grid is below the bars
#     # Adding labels to each bar with adjusted font size
#     for p in ax.patches:
#         height = p.get_height()
#         if height < 0:  # Check if the bar's value is negative
#             vertical_alignment = 'top'
#             vertical_offset = -4
#         else:
#             vertical_alignment = 'center'
#             vertical_offset = 9
    
#         ax.annotate(format(height, '.0f'),
#                     (p.get_x() + p.get_width() / 2., height),
#                     ha='center', 
#                     va=vertical_alignment,
#                     xytext=(0, vertical_offset),
#                     textcoords='offset points',
#                     fontsize=10)
#     plt.tight_layout()
#     filename = f"{obj}_Costs_Day_Resolution.png".replace(" ", "_")
#     plt.savefig(os.path.join(filepath, filename))
#     plt.show()
for df, n in zip(dfs, name):
    sorted_costs = df.sort_index()
    sorted_costs_df = sorted_costs.reset_index()
    sorted_costs_df.columns = ['Index', 'Cost']
    sorted_costs_df["objective"] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[0])
    sorted_costs_df['Scenario'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[2])
    sorted_costs_df['Day'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[3])
    sorted_costs_df['Resolution'] = sorted_costs_df['Index'].apply(lambda x: '_'.join(x.split('_')[4:6]))
    sorted_costs_df['Day_Resolution'] = sorted_costs_df.apply(lambda x: f"Day {x['Day']}:\n {x['Resolution']}", axis=1)
    sorted_costs_df['Scenario_Day'] = sorted_costs_df.apply(lambda x: f"{x['Scenario']}: \n Day {x['Day']}", axis=1)
    
    objectives1 = sorted_costs_df['objective'].unique()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    for obj, ax, ax_title in zip(objectives1, [ax1, ax3], ['Cost', 'Import']):
        sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj) & (sorted_costs_df['Scenario'] != 's1')], 
                    x='Scenario_Day', 
                    y='Cost', 
                    hue='Resolution', 
                    ci=None, 
                    palette='rocket', 
                    dodge=True,
                    ax=ax)
        
        ax.set_title(f'{n} (Objective: {ax_title})', fontsize=16)
        ax.set_ylabel(f'{n} [%]', fontsize=14)
        ax.set_xlabel('Scenario and Day', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(title='Resolution', title_fontsize='13', fontsize='12', loc='upper left')
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        
        for p in ax.patches:
            hatch = hatch_patterns.get(obj, None)
            if hatch:
                p.set_hatch(hatch)
            height = p.get_height()
            if height < 0:
                vertical_alignment = 'top'
                vertical_offset = -4
            else:
                vertical_alignment = 'center'
                vertical_offset = 9
    
            ax.annotate(format(height, '.0f'),
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', 
                         va=vertical_alignment,
                         xytext=(0, vertical_offset),
                         textcoords='offset points',
                         fontsize=10)
    
    for obj, ax, ax_title in zip(objectives1, [ax2, ax4], ['Cost', 'Import']):
        sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj) & (sorted_costs_df['Scenario'] != 's1')],
                    x='Day_Resolution',
                    y='Cost',
                    hue='Scenario',
                    ci=None,
                    palette='viridis',
                    dodge=True,
                    ax=ax)
        
        ax.set_title(f'{n} (Objective: {ax_title})', fontsize=16)
        ax.set_ylabel(f'{n} [%]', fontsize=14)
        ax.set_xlabel('Day and Resolution', fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(title='Scenario', title_fontsize='13', fontsize='12', loc='upper left')
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        
        for p in ax.patches:
            hatch = hatch_patterns.get(obj, None)
            if hatch:
                p.set_hatch(hatch)
            height = p.get_height()
            if height < 0:
                vertical_alignment = 'top'
                vertical_offset = -4
            else:
                vertical_alignment = 'center'
                vertical_offset = 9
    
            ax.annotate(format(height, '.0f'),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', 
                        va=vertical_alignment,
                        xytext=(0, vertical_offset),
                        textcoords='offset points',
                        fontsize=10)
        
    plt.tight_layout()
    
    filename = f"{n}_comparison.png".replace(" ", "_")
    plt.savefig(os.path.join(filepath, filename))
    
    plt.show()
    
#%%    
c= costs.iloc[0]
sorted_costs = c.sort_index()

sorted_costs_df = sorted_costs.reset_index()
sorted_costs_df.columns = ['Index', 'Cost']
sorted_costs_df["objective"] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[0])
sorted_costs_df['Scenario'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[2])
sorted_costs_df['Day'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[3])
sorted_costs_df['Resolution'] = sorted_costs_df['Index'].apply(lambda x: x.split('_')[4])
sorted_costs_df['Day_Resolution'] = sorted_costs_df.apply(lambda x: f"Day {x['Day']}:\n {x['Resolution']} min", axis=1)
sorted_costs_df['Scenario_Day'] = sorted_costs_df.apply(lambda x: f"{x['Scenario']}: \n Day {x['Day']}", axis=1)
    
objectives1 = sorted_costs_df['objective'].unique()

for obj, obj1 in zip(objectives1, obj_name):
    x =sorted_costs_df['Scenario'][1]

    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(14, 12))

    sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj)],
                x='Day_Resolution',
                y='Cost',
                hue='Scenario',
                ci=None,
                palette='viridis',
                dodge=True,
                ax=ax)
    
    ax.set_title(f'Cost Comparison Across Days And Resolutions For Different Scenarios (Objective: {obj1})', fontsize=16)
    ax.set_ylabel(f'{n} [%]', fontsize=14)
    ax.set_xlabel('Day and Resolution', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(title='Scenario', title_fontsize='13', fontsize='12', loc='upper left')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    for p in ax.patches:
        hatch = hatch_patterns[obj1]
        p.set_hatch(hatch)
        height = p.get_height()
        if height < 0:
            vertical_alignment = 'top'
            vertical_offset = -4
        else:
            vertical_alignment = 'center'
            vertical_offset = 9
      
        ax.annotate(format(height, '.0f'),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', 
                    va=vertical_alignment,
                    xytext=(0, vertical_offset),
                    textcoords='offset points',
                    fontsize=10)
      
    sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj)], 
                x='Scenario_Day', 
                y='Cost', 
                hue='Resolution', 
                ci=None, 
                palette='rocket', 
                dodge=True,
                ax=ax1)
    
    ax1.set_title(f'Cost Comparison Across Scenarios And Days For Different Resolutions (Objective: {obj1})', fontsize=16)
    ax1.set_ylabel(f'{n} [%]', fontsize=14)
    ax1.set_xlabel('Scenario and Day', fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.legend(title='Resolution', title_fontsize='13', fontsize='12', loc='upper left')
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    
    for p in ax1.patches:
        hatch = hatch_patterns[obj1]
        p.set_hatch(hatch)
        height = p.get_height()
        if height < 0:
            vertical_alignment = 'top'
            vertical_offset = -4
        else:
            vertical_alignment = 'center'
            vertical_offset = 9
      
        ax1.annotate(format(height, '.0f'),
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', 
                     va=vertical_alignment,
                     xytext=(0, vertical_offset),
                     textcoords='offset points',
                     fontsize=10)
      
    plt.tight_layout()
    
    filename1 = f"{obj1}_Day_Resolution_cost.png".replace(" ", "_")
    plt.savefig(os.path.join(filepath, filename1))
    
    filename2 = f"{obj1}_{n}_Scenario_Day_cost.png".replace(" ", "_")
    plt.savefig(os.path.join(filepath, filename2))
    
    plt.show()
#%%
 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

for obj, ax, ax_title in zip(objectives1, [ax1, ax2], obj_name):
    sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj)],
                x='Day_Resolution',
                y='Cost',
                hue='Scenario',
                ci=None,
                palette='viridis',
                dodge=True,
                ax=ax)
    
    ax.set_title(f'Objective: {ax_title}', fontsize=16)
    ax.set_ylabel(f'{n} [%]', fontsize=14)
    ax.set_xlabel('Day and Resolution', fontsize=14)
    ax.legend(title='Scenario', title_fontsize='13', fontsize='12', loc='upper left')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    for p in ax.patches:
        hatch = hatch_patterns.get(obj, None) 
        if hatch:
            p.set_hatch(hatch)
        height = p.get_height()
        if height < 0:
            vertical_alignment = 'top'
            vertical_offset = -4
        else:
            vertical_alignment = 'center'
            vertical_offset = 9
      
        ax.annotate(format(height, '.0f'),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', 
                    va=vertical_alignment,
                    xytext=(0, vertical_offset),
                    textcoords='offset points',
                    fontsize=10)

for obj, ax, ax_title in zip(objectives1, [ax3, ax4], obj_name):
    sns.barplot(data=sorted_costs_df[(sorted_costs_df['objective'] == obj)], 
                x='Scenario_Day', 
                y='Cost', 
                hue='Resolution', 
                ci=None, 
                palette='rocket', 
                dodge=True,
                ax=ax)
    
    ax.set_title(f'Objective: {ax_title}', fontsize=16)
    ax.set_ylabel(f'{n} [%]', fontsize=14)
    ax.set_xlabel('Scenario and Day', fontsize=14)
    ax.legend(title='Resolution', title_fontsize='13', fontsize='12', loc='upper left')
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    
    for p in ax.patches:
        hatch = hatch_patterns.get(obj, None)
        if hatch:
            p.set_hatch(hatch)
        height = p.get_height()
        if height < 0:
            vertical_alignment = 'top'
            vertical_offset = -4
        else:
            vertical_alignment = 'center'
            vertical_offset = 9
      
        ax.annotate(format(height, '.0f'),
                     (p.get_x() + p.get_width() / 2., height),
                     ha='center', 
                     va=vertical_alignment,
                     xytext=(0, vertical_offset),
                     textcoords='offset points',
                     fontsize=10)

plt.tight_layout()

filename = f"total_cost_comparison.png".replace(" ", "_")
plt.savefig(os.path.join(filepath, filename))

plt.show()



#%% PLot 
filepath = "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/Figures_overleaf/results/operation"
for obj in objectives:
    for s in scenarios:
        for day, price in zip(days, prices):
            fig, ax = plt.subplots(figsize=(14, 10))
            ax1 = ax.twinx()  
            p = price
            for res in resolutions:
                t_l = dfs_resolutions[obj][s][res][f"temp_lower_{day}"]
                t_u = dfs_resolutions[obj][s][res][f"temp_upper_{day}"]
                t_r = dfs_resolutions[obj][s][res][f"temp_room_{day}"]
        
                ax.plot(t_l.index, t_l, '-o', label=f'$T_l$ {res}')
                ax.plot(t_l.index, t_u, '-^', label=f'$T_u$ {res}')
                
            ax1.plot(p.index, p, '-o', label=f'p')
        
            ax.set_title(f'Temperatures in EWH sections for Day {day} Scenrio {s} Objective {obj}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy Volume (MWh)')
            ax.legend(loc='upper left')
            ax.grid(True)
        
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper right')
        
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
            filename2 = f"Temperatures_{obj}_{s}_{day}.png".replace(" ", "_")
            plt.savefig(os.path.join(filepath, filename2))
            plt.show()
                
    #%%
for obj in objectives:
    for s in scenarios:
        for day, price in zip(days, prices):
            fig, (ax,ax1,ax2) = plt.subplots(3,1,figsize=(18, 10),sharex=True)
            solar_prod = df_i[f"solar_prod_{res}_{day}"].resample("H").sum()/1000
            ax1.plot(solar_prod.index,solar_prod, label=f'Solar Production {res}',color = 'red')
            for res in resolutions:
                
                hp_l = dfs_resolutions[obj][s][res][f"hp_lower_{day}"].resample("H").sum()
                hp_u = dfs_resolutions[obj][s][res][f"hp_upper_{day}"].resample("H").sum()
                solar = dfs_resolutions[obj][s][res][f"solar_{day}"].resample("H").sum()
                E = dfs_resolutions[obj][s][res][f"export_power_{day}"].resample("H").sum()
                t_l = dfs_resolutions[obj][s][res][f"temp_lower_{day}"].resample("H").mean()
                t_u = dfs_resolutions[obj][s][res][f"temp_upper_{day}"].resample("H").mean()
                
                
        
                ax.step(hp_l.index, hp_l/1000, label=f'$HP_l$ day: {day} ({res})',where='post',)
                ax.step(hp_l.index, hp_u/1000, label=f'$HP_u$ day: {day} ({res})',where='post')
                
                ax1.plot(solar.index,solar/1000,label=f"$S_t$ {res}")
                
                ax2.plot(t_l.index, t_l, '-o', label=f'$T_{{l,t}}$ {res}')
                ax2.plot(t_u.index, t_u, '-o', label=f'$T_{{u,t}}$ {res}')

                ax.set_title(f'Energy Data for Day {day} Scenrio {s} obj {obj}')
                ax2.set_xlabel('Time')
                ax.set_ylabel('Energy Volume (kWh)')
                ax.legend(loc='upper left')
                ax.grid(True)
                
                ax1.set_ylabel('Solar Consumption (kWh)')
                
                ax.legend(loc='upper left')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper left')
                
                ax.grid()
                ax1.grid()
                ax2.grid()
        
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
                ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
                
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
                
            filename2 = f"{obj}_{s}_{day}_operation.png".replace(" ", "_")
            plt.savefig(os.path.join(filepath, filename2))
            plt.show()
            



#%%
types = [':','-']
plt.rcParams.update({'font.size': 14})

for obj,n in zip(objectives,obj_name):
    for day, price in zip(days, prices):
        fig, (ax,ax1,ax2) = plt.subplots(3,1,figsize=(18, 10),sharex=True)
        axb = ax.twinx() 
        p = price

        for s,t in zip(scenarios,types):
            if s == "s3":
                continue

            hp_l = dfs_resolutions[obj][s]["5_min"][f"hp_lower_{day}"]*12/1000
            hp_u = dfs_resolutions[obj][s]["5_min"][f"hp_upper_{day}"]*12/1000
            solar = dfs_resolutions[obj][s]["5_min"][f"solar_{day}"]
            I = dfs_resolutions[obj][s]["5_min"][f"import_power_{day}"]
            E = dfs_resolutions[obj][s]["5_min"][f"export_power_{day}"]
            t_l = dfs_resolutions[obj][s]["5_min"][f"temp_lower_{day}"]
            t_u = dfs_resolutions[obj][s]["5_min"][f"temp_upper_{day}"]
            solar_prod = df_i[f"solar_prod_5_min_{day}"].resample("5T").sum()
            f = df_i[f"fixed_load_5_min_{day}"].resample("5T").sum()*12/1000
            axb.plot(price.index,price,label="$C_{{sp,h}}$",color="green",linewidth=3.0)
            ax.plot(f.index,f,t,label = "$L_t$ ",color = 'black')

            dem = hp_l+hp_u+f
            ax2.step(hp_l.index, hp_u,t, label=f'$HP_{{u,t}}$  {s} ',color='blue')
            ax2.step(hp_l.index, hp_l,t, label=f'$HP_{{l,t}}$ {s} ',color='orange')
            

            ax1.plot(t_l.index, t_l,t, label=f'$T_{{l,t}}$ {s}',color = "orange")
            ax1.plot(t_u.index, t_u,t, label=f'$T_{{u,t}}$ {s}',color = 'blue')
            
            #ax2.plot(solar.index,solar,label=f"$S_t$",color = 'red')
            
            #ax2.plot(solar.index,solar_prod,color="lightblue",label = "$E_t$")
            #ax2.plot(solar.index,solar,t,color="red",label =f"$S_t$")
            # ax2.step(hp_l.index, hp_l, label=f'$HP_{{l,t}}$ ',where='post',color='orange')
            # ax2.step(hp_l.index, hp_u, label=f'$HP_{{u,t}}$',where='post',color='blue')
        
            #ax.set_title(f'Energy Data for Day {day} Scenrio {s} obj {obj}')
            ax2.set_xlabel('Time',fontsize=14)
            ax.set_ylabel('Consumption [kW]',fontsize=14)
            axb.set_ylabel('Costs [NOK/kWh] ',fontsize=14)
            ax.legend(loc='upper left')
            ax.grid(True)
            
            ax1.set_ylabel(f'Temperature [\u00B0C] ',fontsize=14)
            ax2.set_ylabel('Consumption [kW]',fontsize=14)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=14)
            
            ax.legend(loc='upper left')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            axb.legend(loc='upper right')
        
    
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            
            #ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
            #axb.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
            

            plt.setp(ax1.yaxis.get_majorticklabels(), fontsize=14)
            plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=14)
           # plt.setp(ax.yaxis.get_majorticklabels(), fontsize=14)
            #plt.setp(axb.yaxis.get_majorticklabels(), fontsize=14)
            ax.grid()
            axb.grid()
            ax1.grid()
            ax2.grid()

            res_n =res.replace("_", " ")
            plt.title(f"Results of ASHPWH operation \n (Day: {day}, Objective: {n}, Resolution: 5 min)",fontsize=14)
            plt.tight_layout()
            filename2 = f"{n}_{s}_{day}_operation.png".replace(" ", "_")
            plt.savefig(os.path.join(filepath, filename2))
            
#%%
hp_sum = {obj: {day: pd.DataFrame() for day in days} for obj in objectives}
hp_l_sum = {obj: {day: pd.DataFrame() for day in days} for obj in objectives}
hp_u_sum = {obj: {day: pd.DataFrame() for day in days} for obj in objectives}
hp_cost= {obj: {day: pd.DataFrame() for day in days} for obj in objectives}

for obj in objectives:
    for day, price in zip(days, prices):
        daily_sums = {}
        daily_costs = {}
        hp_l_sums = {}
        hp_u_sums={}
        hourly_cost = cost_sale_day[day][f"import_cost_{day}"]
        for s in scenarios:
            hp_l = dfs_resolutions[obj][s]["5_min"][f"hp_lower_{day}"].sum()
            hp_u = dfs_resolutions[obj][s]["5_min"][f"hp_upper_{day}"].sum()
            hp_l_h = dfs_resolutions[obj][s]["5_min"][f"hp_lower_{day}"].resample("H").sum()/1000
            hp_u_h = dfs_resolutions[obj][s]["5_min"][f"hp_upper_{day}"].resample("H").sum()/1000
            daily_cost = (hp_l_h + hp_u_h) * hourly_cost

            daily_sums[f"HP {s}"] = hp_l + hp_u
            hp_l_sums[f"$HP_l$ {s}"] = hp_l 
            hp_u_sums[f"$HP_u$ {s}"] = hp_u
            daily_costs[f"{s}"] = daily_cost.values.sum()

            hp_sum[obj][day]= pd.DataFrame(daily_sums,index=["sum_{s}"])
            hp_l_sum[obj][day] = pd.DataFrame(hp_l_sums,index=["l_{s}"])
            hp_u_sum[obj][day] = pd.DataFrame(hp_u_sums,index=["u_{s}"])
            hp_cost[obj][day] = pd.DataFrame(daily_costs,index =[f"cost_{s}"])

fig, axes = plt.subplots(len(objectives), len(scenarios), figsize=(12, 10), sharex=True, sharey=True)
colormaps = ['Accent', 'Set1']
types = ["++","--"]
names = ["Cost", "Import"]

for i, (obj, n,t) in enumerate(zip(objectives, names,types)):
    cmap = plt.get_cmap(colormaps[(i * len(days)) % len(colormaps)])
    for j, day in enumerate(days):
        ax = axes[i, j]
        df = hp_cost[obj][day]
        df.plot(kind='bar', ax=ax, legend=False, cmap="Set2",hatch=t)
        ax.set_title(f'Objective: {n} - Day: {day}',fontsize=14)
        #ax.set_xlabel(f'{j}',fontsize=12)
        ax.set_ylabel('Cost',fontsize=12)
        ax.grid()
        ax.set_xlabel('')
        #ax.set_xticklabels(df.index, rotation=0) 
        ax.set_xticklabels('', rotation=0) 
        ax.set_ylim(24,30)
        ax.tick_params(axis='y',labelsize=12)


handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.75),fontsize=12)


handles, labels = axes[-1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.25),fontsize=12)

plt.tight_layout()
filename2 = f"HP_cost_bar.png".replace(" ", "_")
plt.savefig(os.path.join(filepath, filename2), bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(len(objectives), len(scenarios), figsize=(12, 10), sharex=True, sharey=True)
for i, (obj, n,t) in enumerate(zip(objectives, names,types)):
    cmap = plt.get_cmap(colormaps[(i * len(days)) % len(colormaps)])
    for j, day in enumerate(days):
        ax = axes[i, j]
        df_u= hp_u_sum[obj][day]/1000
        df_l = hp_l_sum[obj][day]/1000
        df_u.plot(kind='bar', ax=ax, legend=False, cmap="viridis",hatch=t)
        df_l.plot(kind='bar', ax=ax, legend=False,cmap="Set1",hatch=t)
        # ax.bar(df_u.index, df_u, label=f'{obj} Upper {day}')
        # ax.bar(df_l.index, df_l, bottom=df_u['value'], label=f'{obj} Lower {day}')
        
        #data.plot(kind='bar', stacked=True, ax=ax, legend=False, cmap="Set1", hatch=t)
        max_value = df_l.max().max()
        ax.axhline(max_value, linestyle="dotted", linewidth=3,color='red')
        df = hp_sum[obj][day]/1000
        #df.plot(kind='bar', ax=ax, legend=False, cmap="Set1",hatch=t)
        ax.set_title(f'Objective: {n} - Day: {day}',fontsize=14)
        #ax.set_xlabel(f'{j}',fontsize=12)
        ax.set_ylabel('kWh',fontsize=12)
        ax.grid()
        ax.set_xlabel('')
        #ax.set_xticklabels(df.index, rotation=0) 
        ax.set_xticklabels('', rotation=0) 
        #ax.set_ylim(18,25)
        ax.tick_params(axis='y',labelsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.75),fontsize=14)

handles, labels = axes[-1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.25),fontsize=12)

plt.tight_layout()
filename2 = f"HP_consumption_bar.png".replace(" ", "_")
plt.savefig(os.path.join(filepath, filename2), bbox_inches='tight')
plt.show()

#%%

shifted = {obj:{day:pd.DataFrame() for day in days} for obj in objectives}

for obj,n in zip(objectives,names):
    for day, price in zip(days, prices):
        p = price
        hp_l_1 = dfs_resolutions[obj]["s1"]["5_min"][f"hp_lower_{day}"]*12
        hp_u_1 = dfs_resolutions[obj]["s1"]["5_min"][f"hp_upper_{day}"]*12
        hp_l_2 = dfs_resolutions[obj]["s2"]["5_min"][f"hp_lower_{day}"]*12
        hp_u_2 = dfs_resolutions[obj]["s2"]["5_min"][f"hp_upper_{day}"]*12
        hp_l_3= dfs_resolutions[obj]["s3"]["5_min"][f"hp_lower_{day}"]*12
        hp_u_3 = dfs_resolutions[obj]["s3"]["5_min"][f"hp_upper_{day}"]*12
        q_l_2 = dfs_resolutions[obj]["s2"]["5_min"][f"heat_in_lower_{day}"]*12
        q_u_2 = dfs_resolutions[obj]["s2"]["5_min"][f"heat_in_upper_{day}"]*12
        q_l_3= dfs_resolutions[obj]["s3"]["5_min"][f"heat_in_lower_{day}"]*12
        q_u_3 = dfs_resolutions[obj]["s3"]["5_min"][f"heat_in_upper_{day}"]*12
        temp_l_2 = dfs_resolutions[obj]["s2"]["5_min"][f"temp_lower_{day}"]
        temp_u_2 = dfs_resolutions[obj]["s2"]["5_min"][f"temp_upper_{day}"]
        temp_l_3 = dfs_resolutions[obj]["s3"]["5_min"][f"temp_lower_{day}"]
        temp_u_3 = dfs_resolutions[obj]["s3"]["5_min"][f"temp_upper_{day}"]
        s_2 = dfs_resolutions[obj]["s2"]["5_min"][f"solar_{day}"]*12
        s_3 = dfs_resolutions[obj]["s3"]["5_min"][f"solar_{day}"]*12
        i_2 = dfs_resolutions[obj]["s2"]["5_min"][f"import_power_{day}"]*12
        i_3 = dfs_resolutions[obj]["s3"]["5_min"][f"import_power_{day}"]*12
        
        solar_prod = df_i[f"solar_prod_5_min_{day}"].resample("5T").sum()*12
        f = df_i[f"fixed_load_5_min_{day}"].resample("5T").sum()*12/1000
        t_amb = df_i[f"mean_value_5_min_{day}"].resample("5T").mean()
        
        f_sum = f.resample("H").sum()
        q_l_2_sum =q_l_2.resample("H").sum()
        q_l_3_sum = q_l_3.resample("H").sum()
        q_u_2_sum =q_u_2.resample("H").sum()
        q_u_3_sum = q_u_3.resample("H").sum()
        temp_r_2 = dfs_resolutions[obj]["s2"]["5_min"][f"temp_room_{day}"]
        temp_r_3 = dfs_resolutions[obj]["s3"]["5_min"][f"temp_room_{day}"]
        
        solar_prod_positive = solar_prod > 0
        hp_u_2_solar = hp_u_2[solar_prod_positive]/12
        hp_u_3_solar = hp_u_3[solar_prod_positive]/12
        q_u_2_solar = q_u_2[solar_prod_positive]/12
        q_u_3_solar = q_u_3[solar_prod_positive]/12
        T_u_2_solar =  temp_u_2[solar_prod_positive]
        T_u_3_solar =  temp_u_3[solar_prod_positive]
        
        hp_l_2_solar = hp_l_2[solar_prod_positive]/12
        hp_l_3_solar = hp_l_3[solar_prod_positive]/12
        q_l_2_solar = q_l_2[solar_prod_positive]/12
        q_l_3_solar = q_l_3[solar_prod_positive]/12
        T_l_2_solar =  temp_l_2[solar_prod_positive]
        T_l_3_solar =  temp_l_3[solar_prod_positive]
        
        
        energy_used_s2 = hp_u_2_solar.sum()
        energy_used_s3 = hp_u_3_solar.sum()
        
        shifted_energy_3 = energy_used_s3/hp_u_3.sum()
        shifted_energy_2 = energy_used_s2/hp_u_2.sum()
        #shift = shifted_energy_3-shifted_energy_2
        shift = energy_used_s3-energy_used_s2
        shifted[obj][day] = shift
        
        

        fig,(ax,ax1,axb) = plt.subplots(3,1,figsize=(18,10))

        ax4=axb.twinx()
        ax4.plot(p.index, p, color="green", label="$C_{sp}$")
        axb.plot(i_2.index,i_2/1000, label = "$I$ s2")
        axb.plot(i_2.index,i_3/1000, label = "$I$ s3")
        ax4.set_ylabel('Price [NOK/kWh]', fontsize=12)
        axb.set_ylabel("Consumption [kW]", fontsize=12)
        axb.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axb.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(axb.xaxis.get_majorticklabels(), rotation=0, fontsize=12)
        axb.grid()
        axb.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.05, 0.80))
        ax4.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.05, 1))

        ax.plot(hp_u_3_solar.index, q_u_3_solar * 12 / 1000, label='$Q_{u,in}$ s3 (SP > 0)', color='purple', linewidth=3)
        ax.plot(hp_u_2_solar.index, -q_u_2_solar * 12 / 1000, label='-$Q_{u,in}$ s2 (SP > 0)', color='red', linewidth=3)
        ax.plot(hp_u_3.index, q_u_3 / 1000, label='$Q_{u,in}$ s3', color='blue', linestyle='--')
        ax.plot(hp_u_2.index, -q_u_2 / 1000, label='-$Q_{u,in}$ s2', color='red', linestyle='--')
        
        ax.plot(hp_l_3_solar.index, q_l_3_solar * 12 / 1000, label='$Q_{l,in}$ s3 (SP > 0)', color='purple', linewidth=3)
        ax.plot(hp_l_2_solar.index, -q_l_2_solar * 12 / 1000, label='-$Q_{l,in}$ s2 (SP > 0)', color='red', linewidth=3)
        ax.plot(hp_l_3.index, q_l_3 / 1000, label='$Q_{l,in}$ s3', color='blue', linestyle='--')
        ax.plot(hp_l_2.index, -q_l_2 / 1000, label='-$Q_{l,in}$ s2', color='red', linestyle='--')
        
        ax.grid()
        ax.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))
        ax.set_title(f'ASHPHW Operation Comparison Between Strategy 2 and 3 for Day {day} and Objective: {n}', fontsize=14)
        ax.set_ylabel("Thermal energy [kWh]", fontsize=12)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=12)

        ax1.plot(temp_u_2.index, temp_u_2, label='$T_u$ s2', color='red', linestyle='--')
        ax1.plot(temp_u_3.index, temp_u_3, label='$T_u$ s3', color='blue', linestyle='--')
        ax1.plot(T_u_2_solar.index, T_u_3_solar, label='$T_u$ s2 (SP > 0)', color='purple', linewidth=3)
        ax1.plot(T_u_3_solar.index, T_u_2_solar, label='$T_u$ s3 (SP > 0)', color='red', linewidth=3)
        ax1.plot(temp_l_2.index, temp_l_2, label='$T_l$ s2', color='red', linestyle='--')
        ax1.plot(temp_l_3.index, temp_l_3, label='$T_l$ s3', color='blue', linestyle='--')
        ax1.plot(T_l_2_solar.index, T_l_3_solar, label='$T_l$ s2 (SP > 0)', color='purple', linewidth=3)
        ax1.plot(T_l_3_solar.index, T_l_2_solar, label='$T_l$ s3 (SP > 0)', color='red', linewidth=3)
        ax1.plot(temp_r_2.index,temp_r_2, label="$T_r s2$")
        ax1.plot(temp_r_3.index,temp_r_3, label="$T_r s3$")
        
        ax1.set_ylabel('Temperature [°C]', fontsize=12)
        axb.set_xlabel('Time', fontsize=12)
        ax1.grid()
        ax1.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, fontsize=12)
    
        
        ax.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        axb.tick_params(axis='y', labelsize=12)

        
        plt.tight_layout()
        filename2 = f"shifted_{day}_{obj}.png".replace(" ", "_")
        plt.savefig(os.path.join(filepath, filename2), bbox_inches='tight')
        plt.show()
        

#%%
fig, axes = plt.subplots(len(objectives), len(scenarios), figsize=(12, 10), sharex=True, sharey=True)
for i, (obj, n,t) in enumerate(zip(objectives, names,types)):
    cmap = plt.get_cmap(colormaps[(i * len(days)) % len(colormaps)])
    for j, day in enumerate(days):
        ax = axes[i, j]
        df_u= pd.DataFrame(shifted[obj],index = days)
        df_u[day].plot(kind='bar', ax=ax, legend=False, cmap="viridis",hatch=t)

        ax.set_title(f'Objective: {n} - Day: {day}',fontsize=14)
        #ax.set_xlabel(f'{j}',fontsize=12)
        ax.set_ylabel('kWh',fontsize=12)
        ax.set_xlabel('')  
        #ax.set_xticklabels(df.index, rotation=0) 
        ax.set_xticklabels('', rotation=0) 
        #ax.set_ylim(18,25)
        ax.tick_params(axis='y',labelsize=12)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.75),fontsize=12)

handles, labels = axes[-1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.25),fontsize=12)

plt.tight_layout()
filename2 = f"shifted.png".replace(" ", "_")
plt.savefig(os.path.join(filepath, filename2), bbox_inches='tight')
plt.show()
#%%
plt.rcParams.update({'font.size': 14})

for obj in objectives:
    for day in days:
        hp_l_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"hp_lower_{day}"]
        hp_l_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"hp_lower_{day}"]
        hp_l_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"hp_lower_{day}"]
        
        t_l_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"temp_lower_{day}"]
        t_l_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"temp_lower_{day}"]
        t_l_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"temp_lower_{day}"]
        
        t_r_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"temp_room_{day}"]
        t_r_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"temp_room_{day}"]
        t_r_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"temp_room_{day}"]
        t_amb = df_i[f"mean_value_5_min_{day}"]

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        ax.grid()
        ax2.grid()
        ax.plot(hp_l_s1.index, hp_l_s1*12/1000, label=" $HP_u$ s1")
        ax.plot(hp_l_s2.index, hp_l_s2*12/1000, label="$HP_u$ s2")
        ax.plot(hp_l_s3.index, hp_l_s3*12/1000, label="$HP_u$ s3")
        ax.fill_between(hp_l_s1.index, hp_l_s1*12/1000, alpha=0.3)
        ax.fill_between(hp_l_s2.index, hp_l_s2*12/1000, alpha=0.3)
        ax.fill_between(hp_l_s3.index, hp_l_s3*12/1000, alpha=0.3)
        ax.set_ylabel(" Power consumption [kW] ")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=14)
        
        ax1 = ax.twinx()
        ax1.plot(t_amb, color="gray", linestyle="--", label="$T_{{amb}}$",linewidth=3)
        ax1.set_ylabel("Temperature [°C]")
        ax1.legend(loc="upper right")

        ax2.plot(t_r_s1.index, t_r_s1, label="$T_r$ s1", linestyle="-")
        ax2.plot(t_r_s2.index, t_r_s2, label="$T_r$ s2", linestyle="--")
        ax2.plot(t_r_s3.index, t_r_s3, label="$T_r$ s3", linestyle="-.")
        ax2.plot(t_l_s1.index, t_l_s1, label="$T_l$ s1", linestyle="-")
        ax2.plot(t_l_s2.index, t_l_s2, label="$T_l$ s2", linestyle="--")
        ax2.plot(t_l_s3.index, t_l_s3, label="$T_l$ s3", linestyle="-.")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Temperature [°C]")
        ax2.legend(loc="upper left")
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, fontsize=14)
      

        plt.tight_layout()
        filename1 = f"{obj}_{n}_{day}_hp_lower_operation.png".replace(" ", "_")
        plt.savefig(os.path.join(filepath, filename1))
        plt.show()

for obj in objectives:
    for day in days:
        hp_u_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"hp_upper_{day}"]
        hp_u_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"hp_upper_{day}"]
        hp_u_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"hp_upper_{day}"]
        
        t_u_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"temp_upper_{day}"]
        t_u_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"temp_upper_{day}"]
        t_u_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"temp_upper_{day}"]
        
        t_r_s1 = dfs_resolutions[f"{obj}"]["s1"]["5_min"][f"temp_room_{day}"]
        t_r_s2 = dfs_resolutions[f"{obj}"]["s2"]["5_min"][f"temp_room_{day}"]
        t_r_s3 = dfs_resolutions[f"{obj}"]["s3"]["5_min"][f"temp_room_{day}"]
        t_amb = df_i[f"mean_value_5_min_{day}"]

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        ax.grid()
        ax2.grid()
        ax.plot(hp_u_s1.index, hp_u_s1*12/1000, label="$HP_u$ S1")
        ax.plot(hp_u_s2.index, hp_u_s2*12/1000, label="$HP_u$ S2")
        ax.plot(hp_u_s3.index, hp_u_s3*12/1000, label="$HP_u$ S3")
        ax.fill_between(hp_u_s1.index, hp_u_s1*12/1000, alpha=0.3)
        ax.fill_between(hp_u_s2.index, hp_u_s2*12/1000, alpha=0.3)
        ax.fill_between(hp_u_s3.index, hp_u_s3*12/1000, alpha=0.3)
        ax.set_ylabel("Power consumption [1kW]")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=14)
        
        ax1 = ax.twinx()
        ax1.plot(t_amb, color="gray", linestyle="--", label="$T_{{amb}}$",linewidth=3)
        ax1.set_ylabel("Temperature [°C]")
        ax1.legend(loc="upper right")

        ax2.plot(t_r_s1.index, t_r_s1, label="$T_r$ s1", linestyle="-",linewidth=3)
        ax2.plot(t_r_s2.index, t_r_s2, label="$T_r$ s2", linestyle="--",linewidth=3)
        ax2.plot(t_r_s3.index, t_r_s3, label="$T_r$ s3", linestyle="-.",linewidth=3)
        ax2.plot(t_u_s1.index, t_u_s1, label="$T_l$ s1", linestyle="-",linewidth=3)
        ax2.plot(t_u_s2.index, t_u_s2, label="$T_l$ s2", linestyle="--",linewidth=3)
        ax2.plot(t_u_s3.index, t_u_s3, label="$T_l$ s3", linestyle="-.",linewidth=3)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Temperature [°C]")
        ax2.legend(loc="upper left")
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, fontsize=14)
        
        plt.tight_layout()
        filename1 = f"{obj}_{n}_{day}_hp_upper_operation.png".replace(" ", "_")
        plt.savefig(os.path.join(filepath, filename1))
        plt.show()
        
