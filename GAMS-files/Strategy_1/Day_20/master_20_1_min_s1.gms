Sets
    t "Time intervals" / t1*t1440 /    
    h "Hourly intervals" / h1*h24 /  
;


Parameter solar_prod(t)/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/1_min_data/solar_prod_1_min_data_20_gams.tsv"
/
;
Parameter price(h)
/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/prices_april_20_gams.tsv"
/
;
Parameter utetemp(t)/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/1_min_data/mean_value_1_min_data_20_gams.tsv"
/
;
Parameter fixedload(t)
/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/1_min_data/fixed_load_1_min_data_20_gams.tsv"
/
;
Parameter shower(t)
/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/1_min_data/shower_1_min_data_gams.tsv"
/
;
Parameter fauset(t)
/
$include "C:/Users/eirik/OneDrive/Skrivebord/Master/Masteroppgave/data/1_min_data/fauset_1_min_data_gams.tsv"
/
;


Scalars
bigM /1000/
smallM /0/
d /595E-3/ 
hight /2031E-3 /
V_lower / 136 /
V_upper  / 224  /



T_amb  /20  /
T_ref / 10 /
T_target /60/
T_tank /70 /
q_loss / 50 /

k /2/

temp_upper_min /50/
temp_upper_max / 60 /
temp_upper_max_sol /90/
temp_lower_min_5 /35/
temp_lower_max_5 /45/
temp_lower_min_18 /20/
temp_lower_max_18 /30/
temp_room_min  /18/
temp_room_max  /22/

W /10/
S /15/
L /2.4/

A_window /2/  
U_walls /0.18/  
U_windows /0.8 / 
U_floor /0.1/  
U_roof /0.13/  

P_max /15E3/
P_hp_max/4.5E3/
P_hp_min /2.5E3/

n_floor /0.8/
time /60/

C_water /1.162/
C_air / 0.279/ 
dens_air /1.225/ 

u_lower / 5.186894014962182/
u_upper /1.105691487557002/

A_front_back
A_side
A_floor
A_roof
V
mass_room
mass_upper
mass_lower
h_lower
h_upper
A_lower
A_upper

*tariffs
    paaslag /4.99/
    energiledd_hverdag /48.25/
    energiledd_natt_og_helg /40.75/
    energiledd_pluss_produksjon /5.00/
    mva /0.25/



;

mass_upper =V_upper;
mass_lower =V_lower;


A_front_back = (W * L - A_window) * 2;
A_side = (S * L) * 2;
A_floor = W * S;
A_roof = A_floor;
V = W*S*L;

mass_room = V*dens_air;

h_lower = hight*V_lower/(V_lower+V_upper);
h_upper = hight*V_upper/(V_lower+V_upper);

A_lower = pi*d*h_lower;
A_upper = pi*d*h_upper;



Variables
    import_power(t)
    solar(t)
    hp_lower(t)
    export_power(t)
    heat_in_lower(t)
    lower_loss(t)
    upper_loss(t)
    temp_lower(t)
    temp_upper(t)
    temp_room(t)
    
    s_upper(t)
    s_lower(t)
    s_room(t)
    
    heat_in_upper(t)
    heat_in_room(t)
    room_loss(t)
    heat_lower(t)
    heat_upper(t)
    heat_room(t)
    total_cost
    
    import_energy(h)
    export_energy(h)
    
    temp_diff(t)
    temp_diff_sum
    obj
;
    

Binary Variables
    hp_lower_on(t)
    hp_upper_on(t)
    heat_room_on(t);

Positive Variables
    import_power, export_power, solar, hp_lower, hp_upper,s_upper,heat_in_room;


Equations
    cost_objective
    power_balance_c(t)
    max_export_c(t)
    max_hp_lower_c(t)
    max_hp_upper_c(t)
    one_mode_c(t)
    max_heat_in_lower_c(t)
    max_heat_in_upper_c(t)
    usage_c(t)
    heat_loss_lower_c(t)
    heat_loss_upper_c(t)
    heat_loss_room_c(t)
    energy_lower_c(t)
    energy_upper_c(t)
    energy_room_c(t)
    heatbalance_lower_c(t)
    heatbalance_upper_c(t)
    heatbalance_room_c(t)
    

    import_power_to_energy(h)
    export_power_to_energy(h)
    
    temp_diff_c(t)
    temp_diff_sum_c
    
    weighted_obj
    temp_upper_slack_min(t)
    temp_upper_slack_max(t)
    temp_room_slack_min(t)
    temp_room_slack_max(t)
    
    temp_lower_slack_min_5(t)
    temp_lower_slack_min_5_18(t)
    temp_lower_slack_min_18(t)
    temp_lower_slack_max_5(t)
    temp_lower_slack_max_5_18(t)
    temp_lower_slack_max_18(t)
    slack_max(t)
    temp_room_l(t)
    temp_upper_l(t)
    temp_lower_l(t)
    
    heat_in_room_max(t)
    heat_in_room_min(t)
;

* Constraint definitions

Parameters
    map_t_to_h(t,h)
    spot_pris(h)
    
    avgift_import(h)
    energiledd(h)
    cost(h)
    import_cost(h)
    export_sale(h)
    stotte
    cost
    energiledd
    nett_import(h)
    heat_out_upper(t)
    cop_u(t)
    cop_l(t)
    ;

cop_u(t)= 0.075 * utetemp(t) + 2.125;
cop_l(t) = 0.075 * utetemp(t) + 3.125;

heat_in_room_min(t)..heat_in_room(t) =G= smallM *heat_room_on(t);
heat_in_room_max(t)..heat_in_room(t) =L=bigM*3/time * heat_room_on(t);

map_t_to_h(t,h)$(ord(h) = ceil(ord(t) / time)) = 1;

temp_lower.l('t1') = (temp_lower_min_5 + temp_lower_max_5) / 2;
temp_upper.l('t1') = (temp_upper_min + temp_upper_max) / 2;
temp_room.l('t1') = (temp_room_min + temp_room_max) / 2;


temp_upper_l(t)$(ord(t) = 1)..
                        temp_upper(t) =G= (temp_upper_min + temp_upper_max) / 2;
                        
temp_lower_l(t)$ ( (ord(t) = 1))..
                        temp_lower(t) =G= (temp_lower_min_5 + temp_lower_max_5) / 2;
;

temp_room_l(t)$(ord(t) = 1)..
                        temp_room(t) =G= (temp_room_min + temp_room_max) / 2;

temp_upper_slack_min(t)$(ord(t) > 1)..
                        temp_upper(t) =G= temp_upper_min
- s_upper(t)
;



slack_max(t).. s_upper(t) =L= 40;

temp_upper_slack_max(t)$(ord(t) > 1)..
                        temp_upper(t) =L= temp_upper_max
*+ s_upper(t)
;

temp_room_slack_min(t)$(ord(t) > 1)..
                        temp_room(t) =G= temp_room_min
*- s_room(t)
;

temp_room_slack_max(t)$(ord(t) > 1).. 
                        temp_room(t) =L= temp_room_max
*+ s_room(t)
;

temp_lower_slack_min_5(t)$ (utetemp(t) < -5 and (ord(t) > 1))..
                        temp_lower(t) =G= temp_lower_min_5
*- s_lower(t)
;

temp_lower_slack_min_5_18(t) $ (utetemp(t) >=-5 and utetemp(t) <=18 and (ord(t) > 1))..
                        temp_lower(t)=G= temp_lower_min_5 - 15/23*utetemp(t)
*- s_lower(t)
;
                                            
temp_lower_slack_min_18(t) $ (utetemp(t) > 18 and (ord(t) > 1))..
                        temp_lower(t) =G= temp_lower_min_18
*- s_lower(t)
;


temp_lower_slack_max_5(t)$ (utetemp(t) < -5 and (ord(t) > 1))..
                        temp_lower(t) =L= temp_lower_max_5
                        
*- s_lower(t)
;

temp_lower_slack_max_5_18(t) $ (utetemp(t) >=-5 and utetemp(t) <=18 and (ord(t) > 1))..
                        temp_lower(t) =L= temp_lower_max_5 - 15/23*utetemp(t)
                        
*- s_lower(t)
;
                                            
temp_lower_slack_max_18(t) $ (utetemp(t) > 18 and (ord(t) > 1))..
                        temp_lower(t) =L= temp_lower_max_18 
                        
*- s_lower(t)
;




solar.up(t) = solar_prod(t);
import_power.up(t) = P_max;

loop(t,
    heat_out_upper(t) = shower(t) + fauset(t);
);



*tariff
spot_pris(h) = price(h)*100*1000;

energiledd(h) = energiledd_natt_og_helg;
energiledd(h) $ ( 5 < ord(h) and ord(h) < 21) = energiledd_hverdag;

stotte(h) = 0;
stotte(h) $ (spot_pris(h) > 73) = (spot_pris(h) - 73) * 0.9;
    
import_cost(h) = ((spot_pris(h) - stotte(h))*(1+mva) + paaslag + energiledd(h))/100;
    
export_sale(h) = (spot_pris(h) + energiledd_pluss_produksjon)/100;










temp_diff.l(t) = 0;
*equations

cost_objective..   total_cost =e= sum(h, import_cost(h) * import_energy(h)/1000 - export_sale(h) * export_energy(h)/1000);


import_power_to_energy(h).. import_energy(h) =E=sum(t$(map_t_to_h(t,h)), import_power(t));

export_power_to_energy(h).. export_energy(h) =E=sum(t$(map_t_to_h(t,h)), export_power(t));


power_balance_c(t).. import_power(t) + solar(t)=e= hp_lower(t) + hp_upper(t);



max_export_c(t).. export_power(t) =e= solar_prod(t) - solar(t);


max_hp_lower_c(t).. 
    hp_lower(t) =e= hp_lower_on(t) * P_hp_min/time ;

max_hp_upper_c(t).. 
    hp_upper(t) =e= hp_upper_on(t) * P_hp_max/time;


one_mode_c(t).. 
    hp_lower_on(t) + hp_upper_on(t) =l= 1;


max_heat_in_lower_c(t).. 
    heat_in_lower(t) =e= cop_l(t) * hp_lower(t);


max_heat_in_upper_c(t).. 
    heat_in_upper(t) =e= cop_u(t) * hp_upper(t);




usage_c(t)$((shower(t) > 0) or (fauset(t) > 0)).. 
    heat_in_upper(t) =e= 0;
    


* Heat loss for lower and upper tanks
heat_loss_lower_c(t).. 
    lower_loss(t) =e= u_lower * A_lower * (temp_lower(t) - T_amb) / time;

heat_loss_upper_c(t).. 
    upper_loss(t) =e= u_upper * A_upper * (temp_upper(t) - T_amb) / time;

* Heat loss for the room
heat_loss_room_c(t).. 
    room_loss(t) =e=   100 * ((temp_room(t) + 273) - (utetemp(t) + 273)) / time;


energy_lower_c(t).. 
    heat_lower(t) =e=  (temp_lower(t)-T_ref) * (mass_lower * C_water);

energy_upper_c(t).. 
    heat_upper(t) =e=  (temp_upper(t)-T_ref) * (mass_upper * C_water);

energy_room_c(t).. 
    heat_room(t) =e=  (temp_room(t))* 526;
*(mass_room * C_air);

heatbalance_lower_c(t)$(ord(t) > 1).. 
    heat_lower(t) =e= heat_lower(t-1) + (heat_in_lower(t) - lower_loss(t) - 900/time*heat_room_on(t));
    
heatbalance_upper_c(t)$(ord(t) > 1).. 
    heat_upper(t) =e= heat_upper(t-1) + (heat_in_upper(t) - heat_out_upper(t) - upper_loss(t));

heatbalance_room_c(t)$(ord(t) > 1)..
    heat_room(t) =e= heat_room(t-1) + (heat_in_room(t)) - room_loss(t);
    
temp_diff_c(t)..
    temp_diff(t) =e= k*(T_target-temp_upper(t));
    
temp_diff_sum_c..
    temp_diff_sum =e= sum(t,temp_diff(t));

weighted_obj..
    obj =e= total_cost
 + sum(t,s_upper(t))/5    
*+ sum(t,s_upper(t)+s_room(t)+s_lower(t))
*+sum(t,hp_lower_on(t)+hp_upper_on(t))
*+ temp_diff_sum
    ;

Model heat_pump_system /all/;

heat_pump_system.optcr = 0.02;
Solve heat_pump_system using mip minimizing obj;

execute_unload 'results_master_20_1_min_s1_total_cost.gdx'
    import_power,
    solar,
    hp_lower,
    hp_upper,
    export_power,
    hp_lower_on,
    hp_upper_on,
    heat_in_lower,
    heat_out_upper,
    lower_loss,
    upper_loss,
    temp_lower,
    temp_upper,
    temp_room,
    heat_in_upper,
    heat_in_room,
    room_loss,
    heat_lower,
    heat_upper,
    heat_room,
    total_cost,
    import_energy,
    export_energy
    ;
execute 'gdx2sqlite -i results_master_20_1_min_s1_total_cost.gdx -o scenario1/basecase_20/results_master_20_1_min_s1_total_cost.db -fast';
