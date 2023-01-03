# COPPER model for Canada electricity system, national and provincial scale.
# Written by Reza Arjmand, Ph.D candidate at UVic.
# Version 5.0 Jan 2021 New input data

from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import csv
import gc
import time
import sys

start = time.time()
#os.chdir('/home/arjmand/COPPER5')
cwd = '/project/def-mcpher16/ml_shared/COPPER5_CapCost_CTax_PopGrowth_CCS_True/inputs/'
os.makedirs(cwd, exist_ok=True)
os.chdir(cwd)
#ctax = 50
### planning reserve data
reserve_margin = 0.15  ####[CERI 174 p37(21), ReEDS model doc 2018,]
###percent of available hydro that we can upgrade to be able to store energy (retrofit to add pumped hydro storage)
pump_ret_limit = 0.5
downsampling = False
hierarchical = True
test = False
hydro_development = False
autrarky = False
pump_continous = True
emission_limit = False
emission_limit_ref_year = 2017
local_gas_price = True
OBPS_on = True
SMR_CCS = True
GPS = True
CPO = True
min_installed_gas_PHP = False

##### Reading the configuration excel sheet #####


configuration = pd.read_excel(r'COPPER_configuration.xlsx', header=0)
config = dict(zip(list(configuration.iloc[:]['Parameter']), list(configuration.iloc[:]['Value'])))

# Initializing all sets
# pds=list(range(2030,2051,10))
# pds = list(map(str, pds))
pds = ['2030', '2040', '2050']
season = ['winter', 'summer']
capacity_val = pd.read_csv(r'wind_solar_capacity_value.csv', header=None)
header_A = list(capacity_val.iloc[0, :])
header_B = list(capacity_val.iloc[1, :])
del (header_A[0])
del (header_B[0])
ind = list(capacity_val.iloc[:, 0])
del (ind[0])
del (ind[0])
capacity_value = pd.DataFrame(np.array(capacity_val.loc[2:12, 1:5]),
                              columns=pd.MultiIndex.from_tuples(zip(header_A, header_B)), index=ind)

### All regions
ap = ["British Columbia", "Alberta", "Saskatchewan", "Manitoba", "Ontario", "Quebec", "New Brunswick",
      "Newfoundland and Labrador", "Nova Scotia", "Prince Edward Island"]  # all provinces
aba = ["British Columbia.a", "Alberta.a", "Saskatchewan.a", "Manitoba.a", "Ontario.a", "Ontario.b", "Quebec.a",
       "Quebec.b", "New Brunswick.a", "Newfoundland and Labrador.a", "Newfoundland and Labrador.b", "Nova Scotia.a",
       "Prince Edward Island.a"]  # all possible balancing areas

###Just for creatin input files####
aba1 = ['a', 'b']

#### gas PHP installed capacity limit
gas_PHP_installed_limit = dict()
for ABA in aba:
    gas_PHP_installed_limit[ABA] = 0

##we can skip runnig some days using smaple_rat, for example if sample_rate=3 it will run days 1,4,7,10,...
# number of days that we want to run
if downsampling:
    sample_rate = int(config['sample rate'])
    rundaynum = int(config['run day number'])
    rundays = list(range(1, rundaynum + 1, sample_rate))
    cap_cost_alter = (365 / len(rundays))

elif hierarchical:
    if test:
        run_days = pd.read_csv(r'run_days_test.csv', header=None)
    else:
        run_days = pd.read_csv(r'run_days.csv', header=None)
    rundays = list(run_days.values)
    rundays = [int(RD) for RD in rundays]
    cap_cost_alter = (365 / len(rundays))

#### retrofit cost=0.7 * 201466 (Cost for new PHS)=201466
pump_storage_cost = 141026 / cap_cost_alter
# pump_storage_cost=pump_storage_cost*0.3
store_fix_o_m = 9000 / cap_cost_alter
# store_fix_o_m=store_fix_o_m*0.5

runhours = rundays[-1] * 24
foryear = int(config['forecast year'])
refyear = int(config['refrence year'])

# carbon tax in dollars per tonne co2
ctax = int(config['carbon price'])

GJtoMWh = config['GJtoMWh']  # 3.6
autonomy_pct = config['autonomy_pct']  # 0
pump_hydro_efficiency = config['pump hydro efficiency']  # 0.80
# share carbon reduced
carbon_reduction = config['carbon_reduction']  # 0
# reference case carbon emissions in electricity sector in 2005 in Mt by province (source: https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/provincial-territorial-energy-profiles/provincial-territorial-energy-profiles-explore.html)
carbon_2005_ref = {"British Columbia": 1.04, "Alberta": 48.83, "Saskatchewan": 14.82, "Manitoba": .36, "Ontario": 33.9,
                   "Quebec": 0.65, "New Brunswick": 7.8, "Newfoundland and Labrador": 0.82, "Nova Scotia": 10.77,
                   "Prince Edward Island": 0}

# reference case carbon emissions in electricity sector in 2017 in Mt by province (source: https://www.cer-rec.gc.ca/en/data-analysis/energy-markets/provincial-territorial-energy-profiles/provincial-territorial-energy-profiles-explore.html)
carbon_2017_ref = {"British Columbia": 0.15, "Alberta": 44.33, "Saskatchewan": 15.53, "Manitoba": .07, "Ontario": 1.99,
                   "Quebec": 0.26, "New Brunswick": 3.65, "Newfoundland and Labrador": 1.53, "Nova Scotia": 6.5,
                   "Prince Edward Island": 0.01}
# maximum carbon emissions in forecast year in Mt
carbon_limit = dict()
if emission_limit_ref_year == 2017:
    for AP in ap:
        carbon_limit[AP] = carbon_2017_ref[AP] * (1 - carbon_reduction)

elif emission_limit_ref_year == 2005:
    for AP in ap:
        carbon_limit[AP] = carbon_2005_ref[AP] * (1 - carbon_reduction)

h = list(range(1, 8761))

gltoba = pd.read_csv(r'map_gl_to_ba.csv', header=None)
map_gl_to_ba = dict(gltoba.values)  # map grid locations to balancing areas
del gltoba

gltopr = pd.read_csv(r'map_gl_to_pr.csv', header=None)
map_gl_to_pr = dict(gltopr.values)  # map grid locations to provinces
del gltopr

##### Generation fleets data
if SMR_CCS:
    gendata = pd.read_excel(r'Generation_type_data_SMR_CCS.xlsx', header=0)
else:
    gendata = pd.read_excel(r'Generation_type_data.xlsx', header=0)
allplants = list(gendata.iloc[:]['Type'])

tplants = list()
isthermal = list(gendata.iloc[:]['Is thermal?'])
cc = 0
for i in isthermal:
    if i:
        tplants.append(allplants[cc])
    cc += 1

max_cap_fact = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:][
                                                                'max_cap_fact'])))  # {'gas':0.8, 'peaker': .3, 'nuclear': 0.95, 'coal': 0.9, 'diesel': 0.95, 'waste': 0.9} #annual maximum capacity factor
min_cap_fact = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:][
                                                                'min_cap_fact'])))  # {'gas': 0.2, 'peaker': .02, 'nuclear': 0.75,  'coal': 0.5, 'diesel': 0.05, 'waste': 0.2} #annual minimum capacity factor
ramp_rate_percent = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:][
                                                                     'ramp_rate_percent'])))  # {'gas': 0.1, 'peaker':0.1 , 'nuclear': 0.05,  'coal': 0.05, 'diesel': 0.1, 'waste': 0.05} #ramp rate in percent of capacity per hour
efficiency = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:][
                                                              'efficiency'])))  # {'gas': 0.509, 'peaker': 0.28, 'nuclear': 0.327, 'coal': 0.39, 'diesel': 0.39, 'waste': 0.39}
# fuel_co2=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['fuel_co2'])))#{'gas': .051, 'peaker': .051, 'nuclear': 0, 'coal': .090, 'diesel': .072, 'waste': 0}


fixed_o_m = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:]['fixed_o_m'])))  # (fixedom.values)
for k in fixed_o_m:
    fixed_o_m[k] = fixed_o_m[k] / cap_cost_alter

variable_o_m = dict(
    zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:]['variable_o_m'])))  # dict(variableom.values)

fuelprice = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:]['fuelprice'])))  # dict(fuel_price.values)

capitalcost1 = dict(
    zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:]['capitalcost'])))  # dict(capital_cost.values)
capitalcost = dict()
for ABA in aba:
    for G in capitalcost1:
        capitalcost[ABA + '.' + G] = capitalcost1[G]

if GPS:
    for ABA in aba:
        capitalcost[ABA + '.peaker'] = 1000000000
        capitalcost[ABA + '.coal'] = 1000000000

for k in capitalcost:
    capitalcost[k] = capitalcost[k] / cap_cost_alter

#### Transmission sytem costs
trans_o_m = config['trans_o_m'] / cap_cost_alter  # 10860
transcost = config['transcost'] / cap_cost_alter  # 184
intra_ba_transcost = config['intra_ba_transcost'] / cap_cost_alter  # 557

merrawindsetall = pd.read_csv(r'merra_wind_set_all.csv', header=None)
wl = list(merrawindsetall.iloc[:, 0])

allsolarlocations = pd.read_csv(r'all_solar_locations.csv', header=None)
sl = list(allsolarlocations.iloc[:, 0])

gridlocations = pd.read_csv(r'grid_locations.csv', header=None)
gl = list(gridlocations.iloc[:, 0])
gl = [str(GL) for GL in gl]

distancetogrid = pd.read_csv(r'distance_to_grid.csv', header=None)
distance_to_grid = dict(distancetogrid.values)
windcost = dict()
solarcost = dict()
for GL in gl:
    windcost[GL] = capitalcost[map_gl_to_ba[int(GL)] + '.wind'] + distance_to_grid[int(GL)] * intra_ba_transcost
    solarcost[GL] = capitalcost[map_gl_to_ba[int(GL)] + '.solar'] + distance_to_grid[int(GL)] * intra_ba_transcost

# thermal plant co2 emissions in tonne per MWh of electricity generated
carbondioxide = dict(zip(list(gendata.iloc[:]['Type']), list(gendata.iloc[:]['fuel_co2'])))

#### fuelcost=fuelprice + carbonprice ($/MWh)
if local_gas_price:
    gasprice = {"British Columbia.a": 2.69, "Alberta.a": 2.60, "Saskatchewan.a": 2.55, "Manitoba.a": 2.73,
                "Ontario.a": 6.77, "Ontario.b": 6.77, "Quebec.a": 6.73, "Quebec.b": 6.73, "New Brunswick.a": 6.21,
                "Newfoundland and Labrador.a": 7.39, "Newfoundland and Labrador.b": 7.39, "Nova Scotia.a": 7.39,
                "Prince Edward Island.a": 7.39}
else:
    gasprice = dict()
    for ABA in aba:
        gasprice[ABA] = fuelprice['gas']

if OBPS_on:
    if 'SMR' in tplants:
        OBPS = dict(zip(tplants, [0.37, 0.37, 0.52, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37]))
    else:
        OBPS = dict(zip(tplants, [0.37, 0.52, 0.37, 0.37, 0.37, 0.37]))

    fuelcost = dict()
    for TP in tplants:

        for ABA in aba:
            if TP == 'gas' or TP == 'peaker':
                fuelcost[TP + '.' + ABA] = (gasprice[ABA] / efficiency[TP]) * GJtoMWh
            else:
                fuelcost[TP + '.' + ABA] = (fuelprice[TP] / efficiency[TP]) * GJtoMWh
            if ABA == 'British Columbia.a' or TP == 'waste' or TP == 'SMR' or TP == 'nuclear':
                fuelcost[TP + '.' + ABA] += carbondioxide[TP] * ctax
            else:

                fuelcost[TP + '.' + ABA] += (carbondioxide[TP] - OBPS[TP]) * ctax
else:
    fuelcost = dict()
    for TP in tplants:

        for ABA in aba:
            if TP == 'gas' or TP == 'peaker':
                fuelcost[TP + '.' + ABA] = (gasprice[ABA] / efficiency[TP]) * GJtoMWh
            else:
                fuelcost[TP + '.' + ABA] = (fuelprice[TP] / efficiency[TP]) * GJtoMWh

            fuelcost[TP + '.' + ABA] += carbondioxide[TP] * ctax

    # carbondioxide[TP]=fuel_co2[TP]*GJtoMWh/efficiency[TP]

# demandnodes=pd.read_csv(r'population_locations.csv',header=None) #import demand nodes file
# n=list(demandnodes.iloc[:,0]) #demand nodes

# allnodes=['all'] #dummy for all nodes

# demeqsup=pd.read_csv(r'ba_set.csv',header=None)
# ba=list(demeqsup.iloc[:,0])        #balancing areas where supply equals demand
# del demeqsup


transmapba = pd.read_csv(r'transmission_map_ba.csv', header=None)
transmap = list(transmapba.iloc[:, 0])  # map grid locations to provinces
del transmapba

transmapdis = pd.read_csv(r'transmission_map_distance.csv', header=None)
distance = dict(transmapdis.values)
del transmapdis

# extantgen = pd.read_csv(r'extant_generation_noSiteC.csv',header=None)
# extant_generation=dict(extantgen.values)
# del extantgen

extantwindsolar = pd.read_csv(r'wind_solar_extant2030.csv', header=None)
extant_wind_solar = dict(extantwindsolar.values)
del extantwindsolar

extantcapacity = pd.read_csv(r'extant_capacity.csv', header=0)
extant_capacity = dict()
for PD in pds:
    label_ABA = list(extantcapacity.iloc[:]['ABA'])
    label_ABA = [PD + '.' + LL for LL in label_ABA]
    excap = dict(zip(label_ABA, list(extantcapacity.iloc[:][PD])))
    extant_capacity.update(excap)
# del extantcapacity

extanttrans = pd.read_csv(r'extant_transmission.csv', header=0)
extant_transmission = dict(zip(list(extanttrans.iloc[:]['ABA']), list(extanttrans.iloc[:][str(foryear)])))
del extanttrans

hydrocf = pd.read_csv(r'hydro_cf.csv', header=None)
hydro_cf = dict(hydrocf.values)
del hydrocf

demand_growth = pd.read_csv(r'annual_growth.csv', header=0, index_col=0)
# demand_growth=dict(zip(list(demandgrowth.iloc[:]['AP']),list(demandgrowth.iloc[:][str(foryear)])))
# del demandgrowth

demandall1 = pd.read_csv(r'demand.csv', header=None)
demandall = dict(demandall1.values)
del demandall1

population1 = pd.read_csv(r'population.csv', header=None)
population = dict(population1.values)
del population1

demandus = pd.read_csv(r'us_demand.csv', header=None)
demand_us = dict(demandus.values)
del demandus

maphd = pd.read_csv(r'set_map_days_to_hours.csv', header=None)
map_hd = dict(maphd.values)
del maphd

maphm = pd.read_csv(r'set_map_months_to_hours.csv', header=None)
map_hm = dict(maphm.values)
del maphm

surfacearea = pd.read_csv(r'surface_area.csv', header=None)
surface_area = dict(surfacearea.values)
del surfacearea

demand13to18 = pd.read_csv(r'growth_2013_2018.csv', header=None)
demand_13_to_18 = dict(demand13to18.values)
del demand13to18
# wind_cf = pd.read_csv(r'windcf.csv',header=None)
# windcf=dict(wind_cf.values)

# solar_cf = pd.read_csv(r'solarcf.csv',header=None)
# solarcf=dict(solar_cf.values)

with open('windcf.csv') as csv_file:
    reader = csv.reader(csv_file)
    windcf = dict(reader)

for k in windcf:
    windcf[k] = float(windcf[k])

with open('solarcf.csv') as csv_file:
    reader = csv.reader(csv_file)
    solarcf = dict(reader)

for k in solarcf:
    solarcf[k] = float(solarcf[k])
# windspeed = pd.read_csv(r'merra_wind_all.csv',header=0)
#
# solarcddata = pd.read_csv(r'solar_all.csv',header=0)

## according to the sample_rate these lines remove the days that we don't want to run

demandall_2018 = dict()
demand_all = dict()

national_demand = np.zeros((len(h), len(pds)))
peak_demand = dict()
peak_days = dict()
for PD in pds:
    # peak_demand[SEAS+''+PD]=peak_demand[SEAS]*((1+demand_growth[AP])**(int(PD)-refyear))
    for H in h:
        ND = 0
        for AP in ap:
            demandall_2018[AP + '.' + str(H)] = demandall[AP + '.' + str(H)] * (1 + demand_13_to_18[AP])
            if pds.index(PD) == 0:
                demand_all[PD + '.' + AP + '.' + str(H)] = demandall_2018[AP + '.' + str(H)] * (
                            1 + demand_growth[PD][AP]) ** (int(PD) - refyear)
            else:
                demand_all[PD + '.' + AP + '.' + str(H)] = demand_all[
                                                               pds[pds.index(PD) - 1] + '.' + AP + '.' + str(H)] * (
                        1 + demand_growth[PD][AP]) ** (
                                                                       int(PD) - int(pds[pds.index(PD) - 1]))

            ND += demand_all[PD + '.' + AP + '.' + str(H)]
        national_demand[h.index(H), pds.index(PD)] = ND
    peak_demand[PD + '.' + 'winter'] = max(max(national_demand[:2160, pds.index(PD)]),
                                           max(national_demand[6480:8760, pds.index(PD)]))
    peak_demand[PD + '.' + 'summer'] = max(national_demand[2160:6480, pds.index(PD)])
    peak_days[PD + '.' + 'summer'] = map_hd[int(np.where(national_demand == peak_demand[PD + '.' + 'summer'])[0])]
    peak_days[PD + '.' + 'winter'] = map_hd[int(np.where(national_demand == peak_demand[PD + '.' + 'winter'])[0])]

del h[runhours:]
hours = len(h)

nummonths = map_hm[rundays[-1] * 24]
m = list(range(1, nummonths + 1))

d = rundays.copy()
h3 = h.copy()
for H in h3:
    if map_hd[H] not in rundays:
        h.remove(H)
#        if H in h2:
#            h2.remove(H)
del h3
hours = len(h)
h2 = h.copy()
del h2[hours - 1]

##calculte the diffrence between to hours in a row
time_diff = dict()
for I in list(range(len(h) - 1)):
    time_diff[h[I]] = h[I + 1] - h[I]

gl2 = gl.copy()
for GL in gl2:
    if map_gl_to_pr[int(GL)] not in ap:
        gl.remove(GL)
# windcf=dict()
# solarcf=dict()
# for H in h:
#    for GL in gl:
#        windcf[str(H)+'.'+str(GL)]=0.7
#        solarcf[str(H)+'.'+str(GL)]=solarcddata.iloc[int(H)-1][int(GL)-1]

###maximum wind and solar capacity that can be installed in MW per square km
maxwindperkmsq = config['maxwindperkmsq']  # 2
maxsolarperkmsq = config['maxsolarperkmsq']  # 31.28
maxwind = dict()
maxsolar = dict()
for GL in gl:
    maxwind[GL] = surface_area[int(GL)] * maxwindperkmsq
    maxsolar[GL] = surface_area[int(GL)] * maxsolarperkmsq

pump_hours = 8;
# pump_hydro_capacity={'1508':174}
ba_pump_hydro_capacity = dict()
for ABA in aba:
    ba_pump_hydro_capacity[ABA] = 0
ba_pump_hydro_capacity['Ontario.a'] = 174

translossfixed = config['translossfixed']  # 0.02
translossperkm = config['translossperkm']  # 0.00003
transloss = dict()
for ABA in aba:
    for ABBA in aba:
        if ABA + '.' + ABBA in distance:
            transloss[ABA + '.' + ABBA] = distance[ABA + '.' + ABBA] * translossperkm + translossfixed

# national_demand=list()
# for i in range(1,8761):
#     nd=sum(demandall[AP+'.'+str(i)] for AP in ap )
#     national_demand.append(nd)
# peak_demand=[max(max(national_demand[:2160]),max(national_demand[6480:8760])),max(national_demand[2160:6480])]


populationaba = dict()
for ABA in aba:
    populationaba[ABA] = 0
for ABA in aba:
    for GL in population:
        if map_gl_to_ba[int(GL)] == ABA:
            populationaba[ABA] = populationaba[ABA] + population[GL]
populationap = dict()
demand = dict()

for PD in pds:
    for AP in ap:
        populationap[AP] = sum(populationaba[ABA] for ABA in aba if AP in ABA)
    for ABA in aba:
        pvba = ABA.replace('.a', '')
        pvba = pvba.replace('.b', '')
        for H in h:
            demand[PD + '.' + ABA + '.' + str(H)] = demand_all[PD + '.' + pvba + '.' + str(H)] * (
                        populationaba[ABA] / populationap[pvba])

extant_thermal = dict()
for AP in ap:
    for ABA in aba1:
        for TP in tplants:
            for PD in pds:
                extant_thermal[PD + '.' + AP + '.' + ABA + '.' + TP] = 0
                if PD + '.' + AP + '.' + ABA + '.' + TP in extant_capacity:
                    extant_thermal[PD + '.' + AP + '.' + ABA + '.' + TP] = extant_capacity[
                        PD + '.' + AP + '.' + ABA + '.' + TP]
                if CPO and TP == 'coal' and int(PD) >= 2030:
                    extant_thermal[PD + '.' + AP + '.' + ABA + '.' + TP] = 0
# for GL in gl:
#    for TP in tplants:
#        for EXG in extant_generation:
#            if EXG==str(GL)+'.'+TP:
#                if map_gl_to_ba[int(GL)]+'.'+TP in extant_thermal:
#                    extant_thermal[map_gl_to_ba[int(GL)]+'.'+TP]+=extant_generation[EXG]


hydro_capacity = dict()
extant_wind_gen = dict()
extant_solar_gen = dict()
for PD in pds:
    for AP in ap:
        for ABA in aba1:
            #        hydro_capacity[AP+'.'+ABA+'.'+'hydro']=0
            for H in h:
                extant_wind_gen[PD + '.' + AP + '.' + ABA + '.' + str(H)] = 0
                extant_solar_gen[PD + '.' + AP + '.' + ABA + '.' + str(H)] = 0

# CALCULATE OUTPUT POWER FOR EXTANT WIND FARM AND SOLAR POWER PLANTS
for PD in pds:
    for GL in gl:
        for EXG in extant_wind_solar:
            if int(PD) <= 2030:
                #            if EXG==str(GL)+'.'+'hydro':
                #                hydro_capacity[map_gl_to_ba[int(GL)]+'.'+'hydro']+=extant_generation[EXG]
                if EXG == str(GL) + '.' + 'wind':
                    for H in h:
                        extant_wind_gen[PD + '.' + map_gl_to_ba[int(GL)] + '.' + str(H)] += (
                                    extant_wind_solar[EXG] * windcf[str(H) + '.' + str(GL)])
                if EXG == str(GL) + '.' + 'solar':
                    for H in h:
                        extant_solar_gen[PD + '.' + map_gl_to_ba[int(GL)] + '.' + str(H)] += (
                                    extant_wind_solar[EXG] * solarcf[str(H) + '.' + str(GL)])

## calculate hydro historic for different type of hydro power plant
# ror_share=0.3
# intraday_share=0.5
hydro_minflow = 0.1
# ;hydro=dict()
ror_hydroout = dict()
day_hydroout = dict()
month_hydroout = dict()
day_hydro_historic = dict()
month_hydro_historic = dict()
ror_hydro_capacity = dict()
day_hydro_capacity = dict()
month_hydro_capacity = dict()
day_minflow = dict()
month_minflow = dict()
for PD in pds:
    for AP in ap:
        for ABA in aba1:
            ror_hydro_capacity[PD + '.' + AP + '.' + ABA] = 0
            day_hydro_capacity[PD + '.' + AP + '.' + ABA] = 0
            month_hydro_capacity[PD + '.' + AP + '.' + ABA] = 0
            if PD + '.' + AP + '.' + ABA + '.' + 'hydro_run' in extant_capacity:
                ror_hydro_capacity[PD + '.' + AP + '.' + ABA] = extant_capacity[
                    PD + '.' + AP + '.' + ABA + '.' + 'hydro_run']
                day_hydro_capacity[PD + '.' + AP + '.' + ABA] = extant_capacity[
                    PD + '.' + AP + '.' + ABA + '.' + 'hydro_daily']
                month_hydro_capacity[PD + '.' + AP + '.' + ABA] = extant_capacity[
                    PD + '.' + AP + '.' + ABA + '.' + 'hydro_monthly']
            day_minflow[PD + '.' + AP + '.' + ABA] = day_hydro_capacity[PD + '.' + AP + '.' + ABA] * hydro_minflow
            month_minflow[PD + '.' + AP + '.' + ABA] = month_hydro_capacity[PD + '.' + AP + '.' + ABA] * hydro_minflow
            for D in d:
                day_hydro_historic[PD + '.' + str(D) + '.' + AP + '.' + ABA] = 0

            for M in m:
                month_hydro_historic[PD + '.' + str(M) + '.' + AP + '.' + ABA] = 0
            for H in h:

                #            hydro[str(H)+'.'+AP+'.'+ABA]=0
                #            if AP+'.'+str(H) in hydro_cf:
                #                hydro[str(H)+'.'+AP+'.'+ABA]=hydro_cf[AP+'.'+str(H)]*hydro_capacity[AP+'.'+ABA+'.'+'hydro']
                if AP + '.' + str(H) in hydro_cf:
                    ror_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = ror_hydro_capacity[
                                                                                 PD + '.' + AP + '.' + ABA] * hydro_cf[
                                                                                 AP + '.' + str(H)]
                    day_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = day_hydro_capacity[
                                                                                 PD + '.' + AP + '.' + ABA] * hydro_cf[
                                                                                 AP + '.' + str(H)]
                    month_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = month_hydro_capacity[
                                                                                   PD + '.' + AP + '.' + ABA] * \
                                                                               hydro_cf[AP + '.' + str(H)]
                else:
                    ror_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = 0
                    day_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = 0
                    month_hydroout[PD + '.' + str(H) + '.' + AP + '.' + ABA] = 0

                day_hydro_historic[PD + '.' + str(map_hd[H]) + '.' + AP + '.' + ABA] = day_hydro_historic[
                                                                                           PD + '.' + str(map_hd[
                                                                                                              H]) + '.' + AP + '.' + ABA] + \
                                                                                       day_hydroout[PD + '.' + str(
                                                                                           H) + '.' + AP + '.' + ABA]
                month_hydro_historic[PD + '.' + str(map_hm[H]) + '.' + AP + '.' + ABA] = month_hydro_historic[
                                                                                             PD + '.' + str(map_hm[
                                                                                                                H]) + '.' + AP + '.' + ABA] + \
                                                                                         month_hydroout[PD + '.' + str(
                                                                                             H) + '.' + AP + '.' + ABA]

## hydro renewal and greenfield necessary inputs
if hydro_development:
    if pump_continous:
        hydro_new = pd.read_excel(r'hydro_new_recon_nopump.xlsx', header=0)
    else:
        hydro_new = pd.read_excel(r'hydro_new_recon.xlsx', header=0)

    hydro_renewal = list(hydro_new.iloc[:]['Short Name'])
    cost_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Annualized Capital Cost ($M/year)'])))
    capacity_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Additional Capacity (MW)'])))
    devperiod_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Development Time (years)'])))
    location_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Balancing Area'])))
    distance_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Distance to Grid (km)'])))
    type_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Type'])))
    fixed_o_m_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Fixed O&M ($/MW-year)'])))
    variable_o_m_renewal = dict(zip(hydro_renewal, list(hydro_new.iloc[:]['Variable O&M ($/MWh)'])))

    hr_ror = list()
    cost_ror_renewal = dict()
    capacity_ror_renewal = dict()
    hr_ror_location = dict()

    hr_day = list()
    cost_day_renewal = dict()
    capacity_day_renewal = dict()
    hr_day_location = dict()

    hr_mo = list()
    cost_month_renewal = dict()
    capacity_month_renewal = dict()
    hr_month_location = dict()

    hr_pump = list()
    cost_pump_renewal = dict()
    capacity_pump_renewal = dict()
    hr_pump_location = dict()
    for k in hydro_renewal:
        if foryear - 2020 >= devperiod_renewal[k]:
            if type_renewal[k] == 'hydro_run':
                hr_ror.append(k)
                cost_ror_renewal[k] = cost_renewal[k] * 1000000 + distance_renewal[k] * intra_ba_transcost * \
                                      capacity_renewal[k]
                capacity_ror_renewal[k] = capacity_renewal[k]
                hr_ror_location[k] = location_renewal[k]

            if type_renewal[k] == 'hydro_daily':
                hr_day.append(k)
                cost_day_renewal[k] = cost_renewal[k] * 1000000 + distance_renewal[k] * intra_ba_transcost * \
                                      capacity_renewal[k]
                capacity_day_renewal[k] = capacity_renewal[k]
                hr_day_location[k] = location_renewal[k]

            if type_renewal[k] == 'hydro_monthly':
                hr_mo.append(k)
                cost_month_renewal[k] = cost_renewal[k] * 1000000 + distance_renewal[k] * intra_ba_transcost * \
                                        capacity_renewal[k]
                capacity_month_renewal[k] = capacity_renewal[k]
                hr_month_location[k] = location_renewal[k]
            if type_renewal[k] == 'hydro_pump':
                hr_pump.append(k)
                cost_pump_renewal[k] = cost_renewal[k] * 1000000 + distance_renewal[k] * intra_ba_transcost * \
                                       capacity_renewal[k]
                capacity_pump_renewal[k] = capacity_renewal[k]
                hr_pump_location[k] = location_renewal[k]

    ror_renewalout = dict()
    for HR_ROR in hr_ror:
        for H in h:
            province_loc = hr_ror_location[HR_ROR].replace('.a', '')
            province_loc = province_loc.replace('.b', '')
            ror_renewalout[str(H) + '.' + HR_ROR] = capacity_ror_renewal[HR_ROR] * hydro_cf[province_loc + '.' + str(H)]

    for k in cost_ror_renewal:
        cost_ror_renewal[k] = cost_ror_renewal[k] / cap_cost_alter

    day_renewal_historic = dict()
    day_renewalout = dict()
    for HR_DAY in hr_day:
        for D in d:
            day_renewal_historic[str(D) + '.' + HR_DAY] = 0
        for H in h:
            province_loc = hr_day_location[HR_DAY].replace('.a', '')
            province_loc = province_loc.replace('.b', '')
            day_renewalout[str(H) + '.' + HR_DAY] = capacity_day_renewal[HR_DAY] * hydro_cf[province_loc + '.' + str(H)]
            day_renewal_historic[str(map_hd[H]) + '.' + HR_DAY] = day_renewal_historic[str(map_hd[H]) + '.' + HR_DAY] + \
                                                                  day_renewalout[str(H) + '.' + HR_DAY]

    for k in cost_day_renewal:
        cost_day_renewal[k] = cost_day_renewal[k] / cap_cost_alter

    month_renewal_historic = dict()
    month_renewalout = dict()
    for HR_MO in hr_mo:
        for M in m:
            month_renewal_historic[str(M) + '.' + HR_MO] = 0
        for H in h:
            province_loc = hr_month_location[HR_MO].replace('.a', '')
            province_loc = province_loc.replace('.b', '')
            month_renewalout[str(H) + '.' + HR_MO] = capacity_month_renewal[HR_MO] * hydro_cf[
                province_loc + '.' + str(H)]
            month_renewal_historic[str(map_hm[H]) + '.' + HR_MO] = month_renewal_historic[
                                                                       str(map_hm[H]) + '.' + HR_MO] + month_renewalout[
                                                                       str(H) + '.' + HR_MO]

    for k in cost_month_renewal:
        cost_month_renewal[k] = cost_month_renewal[k] / cap_cost_alter

    for k in cost_pump_renewal:
        cost_pump_renewal[k] = cost_pump_renewal[k] / cap_cost_alter

#####recontract input data ###########33

windcost_recon = config['windcost_recon'] / cap_cost_alter  # 118662.7485
solarcost_recon = config['solarcost_recon'] / cap_cost_alter  # 94204.76075
# column_name_loc=str(foryear)+'_ended'
# column_name_cap=str(foryear)+'_ended_cap'
windsolarrecon = pd.read_csv(r'wind_solar_location_recon.csv', header=0)
wind_solar_recon_2030 = dict(
    zip(list(windsolarrecon.iloc[:]['2030_ended']), list(windsolarrecon.iloc[:]['2030_ended_cap'])))
wind_solar_recon_2050 = dict(
    zip(list(windsolarrecon.iloc[:]['2050_ended']), list(windsolarrecon.iloc[:]['2050_ended_cap'])))

wind_recon_capacity = dict()
solar_recon_capacity = dict()
for PD in pds:
    for GL in gl:
        wind_recon_capacity[PD + '.' + GL] = 0
        solar_recon_capacity[PD + '.' + GL] = 0
        if PD == '2030':
            if str(GL) + '.' + 'wind' in wind_solar_recon_2030:
                wind_recon_capacity[PD + '.' + GL] = wind_solar_recon_2030[str(GL) + '.' + 'wind']
            if str(GL) + '.' + 'solar' in wind_solar_recon_2030:
                solar_recon_capacity[PD + '.' + GL] = wind_solar_recon_2030[str(GL) + '.' + 'solar']

        if PD == '2050':
            if str(GL) + '.' + 'wind' in wind_solar_recon_2030:
                wind_recon_capacity[PD + '.' + GL] = wind_solar_recon_2030[str(GL) + '.' + 'wind']
                if str(GL) + '.' + 'wind' in wind_solar_recon_2050:
                    wind_recon_capacity[PD + '.' + GL] = wind_recon_capacity[PD + '.' + GL] + wind_solar_recon_2050[
                        str(GL) + '.' + 'wind']
            elif str(GL) + '.' + 'wind' in wind_solar_recon_2050:
                wind_recon_capacity[PD + '.' + GL] = wind_solar_recon_2050[str(GL) + '.' + 'wind']

            if str(GL) + '.' + 'solar' in wind_solar_recon_2030:
                solar_recon_capacity[PD + '.' + GL] = wind_solar_recon_2030[str(GL) + '.' + 'solar']
                if str(GL) + '.' + 'solar' in wind_solar_recon_2050:
                    solar_recon_capacity[PD + '.' + GL] = solar_recon_capacity[PD + '.' + GL] + wind_solar_recon_2050[
                        str(GL) + '.' + 'solar']
            elif str(GL) + '.' + 'solar' in wind_solar_recon_2050:
                solar_recon_capacity[PD + '.' + GL] = wind_solar_recon_2050[str(GL) + '.' + 'solar']

# period_steps=3
# pds=list(range(refyear+period_steps,foryear,period_steps))
# if foryear not in Periods:
#     pds.append(foryear)

cleared_data = gc.collect()
######### Doublicate some sets ###############
ttplants = tplants
# nn=n
ggl = gl
hh = h
app = ap
abba = aba

end = time.time()
print(f'\n==================================================\n\
Initializing input data time (Sec): {round((end - start) / 60)} Min and {round((end - start) % 60)} Sec \
\n==================================================')
start = time.time()

########Creating Model######################

model = ConcreteModel()

#### Defineing variables####

model.capacity_therm = Var(pds, aba, tplants, within=NonNegativeReals, initialize=1)  # new thermal plant capacity in MW
model.retire_therm = Var(pds, aba, tplants, within=NonNegativeReals,
                         initialize=0)  # retire extant thermal capacity in MW
model.capacity_wind = Var(pds, gl, within=NonNegativeReals, initialize=0)  # wind plant capacity in MW
model.capacity_solar = Var(pds, gl, within=NonNegativeReals, initialize=0)  # solar plant capacity in MW
if pump_continous:
    model.capacity_storage = Var(pds, aba, within=NonNegativeReals, initialize=0)  # storage plant capacity in MW
model.supply = Var(pds, h, aba, tplants, within=NonNegativeReals, initialize=0)  # fossil fuel supply in MW
model.windout = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # wind hourly power output
model.solarout = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # solar hourly power output
model.pumpout = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # pumped hydro hourly output
model.pumpin = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # pumped hydro hourly input
model.pumpenergy = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # total stored pump hydro energy in MWh
model.daystoragehydroout = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # day storage hydro output in MW
model.monthstoragehydroout = Var(pds, h, aba, within=NonNegativeReals, initialize=0)  # month storage hydro output in MW
model.transmission = Var(pds, h, aba, abba, within=NonNegativeReals,
                         initialize=0)  # hourly transmission in MW from ap,ba to apa,abba
model.capacity_transmission = Var(pds, aba, abba, within=NonNegativeReals,
                                  initialize=0)  # transmission capacity in MW from ap,aba to app,abba
# model.carbon=Var(ap,aba, within=NonNegativeReals,initialize=0)  #carbon emissions annual in Mt
if hydro_development:
    model.ror_renewal_binary = Var(pds, hr_ror, within=Binary, initialize=1)
    model.day_renewal_binary = Var(pds, hr_day, within=Binary, initialize=1)
    model.month_renewal_binary = Var(pds, hr_mo, within=Binary, initialize=1)
    model.dayrenewalout = Var(pds, h, hr_day, within=NonNegativeReals, initialize=0)
    model.monthrenewalout = Var(pds, h, hr_mo, within=NonNegativeReals, initialize=0)
    if not pump_continous:
        model.pumphydro = Var(pds, hr_pump, within=Binary, initialize=0)
model.capacity_wind_recon = Var(pds, gl, within=NonNegativeReals, initialize=0)  # wind recontract capacity in MW
model.capacity_solar_recon = Var(pds, gl, within=NonNegativeReals, initialize=0)  # solar recontract capacity in MW
# model.capitalcost=Var(within=NonNegativeReals)
# model.fuelcost=Var(within=NonNegativeReals)
# model.fixedOM=Var(within=NonNegativeReals)
# model.variableOM=Var(within=NonNegativeReals)
# model.hydrorenewalcost=Var(within=NonNegativeReals)


for ABA in aba:
    for PD in pds:
        model.pumpenergy[PD, h[0], ABA].fix(0)


###Objective function total cost minimization###
def obj_rule(model):
    capcost = sum(
        model.capacity_therm[PD, ABA, TP] * capitalcost[ABA + '.' + TP] * (len(pds) - pds.index(PD)) for PD in pds for
        TP in tplants for ABA in aba) \
              + sum(model.capacity_wind[PD, GL] * windcost[GL] * (len(pds) - pds.index(PD)) for PD in pds for GL in gl) \
              + sum(
        model.capacity_solar[PD, GL] * solarcost[GL] * (len(pds) - pds.index(PD)) for PD in pds for GL in gl) \
              + sum(
        model.capacity_wind_recon[PD, GL] * windcost_recon * (len(pds) - pds.index(PD)) for PD in pds for GL in gl) \
              + sum(
        model.capacity_solar_recon[PD, GL] * solarcost_recon * (len(pds) - pds.index(PD)) for PD in pds for GL in gl) \
              + sum(
        model.capacity_transmission[PD, ABA, ABBA] * transcost * distance[ABA + '.' + ABBA] * (len(pds) - pds.index(PD))
        for PD in pds for ABA in aba for ABBA in aba if ABA + '.' + ABBA in transmap)

    fcost = sum(
        model.supply[PD, H, ABA, TP] * fuelcost[TP + '.' + ABA] for PD in pds for H in h for ABA in aba for TP in
        tplants)

    fixedOM = sum((extant_thermal[pds[0] + '.' + ABA + '.' + TP] + sum(
        model.capacity_therm[PDD, ABA, TP] - model.retire_therm[PDD, ABA, TP] for PDD in pds[:pds.index(PD) + 1]) * (
                               len(pds) - pds.index(PD))) * fixed_o_m[TP] for PD in pds for ABA in aba for TP in
                  tplants) \
              + sum(
        (model.capacity_wind[PD, GL] + model.capacity_wind_recon[PD, GL]) * (len(pds) - pds.index(PD)) * fixed_o_m[
            'wind'] for PD in pds for GL in gl) + sum(
        extant_wind_solar[str(GL) + '.' + 'wind'] * fixed_o_m['wind'] for GL in gl if
        str(GL) + '.' + 'wind' in extant_wind_solar) \
              + sum(
        (model.capacity_solar[PD, GL] + model.capacity_solar_recon[PD, GL]) * (len(pds) - pds.index(PD)) * fixed_o_m[
            'solar'] for PD in pds for GL in gl) + sum(
        extant_wind_solar[str(GL) + '.' + 'solar'] * fixed_o_m['solar'] for GL in gl if
        str(GL) + '.' + 'solar' in extant_wind_solar) \
              + sum(
        model.capacity_transmission[PD, ABA, ABBA] * (len(pds) - pds.index(PD)) * trans_o_m for PD in pds for ABA in aba
        for ABBA in aba if ABA + '.' + ABBA in transmap) \
              + sum(extant_transmission[ABA + '.' + ABBA] * trans_o_m for ABA in aba for ABBA in aba if
                    ABA + '.' + ABBA in extant_transmission) \
              + sum(ror_hydro_capacity[PD + '.' + ABA] * fixed_o_m['hydro'] for PD in pds for ABA in aba) \
              + sum(day_hydro_capacity[PD + '.' + ABA] * fixed_o_m['hydro'] for PD in pds for ABA in aba) \
              + sum(month_hydro_capacity[PD + '.' + ABA] * fixed_o_m['hydro'] for PD in pds for ABA in aba)

    variableOM = sum(
        model.supply[PD, H, ABA, TP] * variable_o_m[TP] for PD in pds for H in h for ABA in aba for TP in tplants) \
                 + sum(model.windout[PD, H, ABA] * variable_o_m['wind'] for PD in pds for H in h for ABA in aba) \
                 + sum(model.solarout[PD, H, ABA] * variable_o_m['solar'] for PD in pds for H in h for ABA in aba) \
                 + sum(
        ror_hydroout[PD + '.' + str(H) + '.' + ABA] * variable_o_m['hydro'] for PD in pds for H in h for ABA in aba) \
                 + sum(
        model.daystoragehydroout[PD, H, ABA] * variable_o_m['hydro'] for PD in pds for H in h for ABA in aba) \
                 + sum(
        model.monthstoragehydroout[PD, H, ABA] * variable_o_m['hydro'] for PD in pds for H in h for ABA in aba)

    hydrorenewalcost = 0
    if hydro_development:
        hydrorenewalcost = sum(
            cost_ror_renewal[HR_ROR] * model.ror_renewal_binary[PD, HR_ROR] * (len(pds) - pds.index(PD)) for PD in pds
            for HR_ROR in hr_ror) \
                           + sum(
            cost_day_renewal[HR_DAY] * model.day_renewal_binary[PD, HR_DAY] * (len(pds) - pds.index(PD)) for PD in pds
            for HR_DAY in hr_day) \
                           + sum(
            cost_month_renewal[HR_MO] * model.month_renewal_binary[PD, HR_MO] * (len(pds) - pds.index(PD)) for PD in pds
            for HR_MO in hr_mo) \
                           + sum(
            capacity_ror_renewal[HR_ROR] * model.ror_renewal_binary[PD, HR_ROR] * (len(pds) - pds.index(PD)) *
            fixed_o_m_renewal[HR_ROR] for PD in pds for HR_ROR in hr_ror) \
                           + sum(
            capacity_day_renewal[HR_DAY] * model.day_renewal_binary[PD, HR_DAY] * (len(pds) - pds.index(PD)) *
            fixed_o_m_renewal[HR_DAY] for PD in pds for HR_DAY in hr_day) \
                           + sum(
            capacity_month_renewal[HR_MO] * model.month_renewal_binary[PD, HR_MO] * (len(pds) - pds.index(PD)) *
            fixed_o_m_renewal[HR_MO] for PD in pds for HR_MO in hr_mo) \
                           + sum(
            ror_renewalout[str(H) + '.' + HR_ROR] * model.ror_renewal_binary[PD, HR_ROR] * variable_o_m_renewal[HR_ROR]
            for PD in pds for H in h for HR_ROR in hr_ror) \
                           + sum(
            model.dayrenewalout[PD, H, HR_DAY] * variable_o_m_renewal[HR_DAY] for PD in pds for H in h for HR_DAY in
            hr_day) \
                           + sum(
            model.monthrenewalout[PD, H, HR_MO] * variable_o_m_renewal[HR_MO] for PD in pds for H in h for HR_MO in
            hr_mo)

        if not pump_continous:
            hydrorenewalcost += sum(
                cost_pump_renewal[HR_PUMP] * model.pumphydro[PD, HR_PUMP] * (len(pds) - pds.index(PD)) for PD in pds for
                HR_PUMP in hr_pump) \
                                + sum(
                capacity_pump_renewal[HR_PUMP] * model.pumphydro[PD, HR_PUMP] * (len(pds) - pds.index(PD)) *
                fixed_o_m_renewal[HR_PUMP] for PD in pds for HR_PUMP in hr_pump)
    newpump_cost = 0
    if pump_continous:
        newpump_cost = sum(
            model.capacity_storage[PD, ABA] * pump_storage_cost * (len(pds) - pds.index(PD)) for PD in pds for ABA in
            aba) \
                       + sum(
            store_fix_o_m * model.capacity_storage[PD, ABA] * (len(pds) - pds.index(PD)) for PD in pds for ABA in aba)

    return (capcost + fcost + variableOM + hydrorenewalcost + newpump_cost + fixedOM)


model.obj = Objective(rule=obj_rule, sense=minimize)


######Planning reserve requirment#####

def planning_reserve(model, PD, SEAS):
    ind = pds.index(PD)
    cap_val = sum((extant_thermal[pds[0] + '.' + ABA + '.' + TP] + sum(
        model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1])) for ABA in aba
                  for TP in tplants) \
              + sum((model.capacity_wind[PD, GL] + model.capacity_wind_recon[PD, GL]) * float(
        capacity_value[SEAS]['wind'][map_gl_to_pr[int(GL)]]) for GL in gl) + sum(
        extant_wind_solar[str(GL) + '.' + 'wind'] * float(capacity_value[SEAS]['wind'][map_gl_to_pr[int(GL)]]) for GL in
        gl if str(GL) + '.' + 'wind' in extant_wind_solar) \
              + sum((model.capacity_solar[PD, GL] + model.capacity_solar_recon[PD, GL]) * float(
        capacity_value[SEAS]['solar'][map_gl_to_pr[int(GL)]]) for GL in gl) + sum(
        extant_wind_solar[str(GL) + '.' + 'solar'] * float(capacity_value[SEAS]['solar'][map_gl_to_pr[int(GL)]]) for GL
        in gl if str(GL) + '.' + 'solar' in extant_wind_solar) \
              + sum(ror_hydro_capacity[PD + '.' + ABA] for ABA in aba) \
              + sum(day_hydro_capacity[PD + '.' + ABA] for ABA in aba) \
              + sum(month_hydro_capacity[PD + '.' + ABA] for ABA in aba) \

    if hydro_development:
        cap_val += sum(capacity_ror_renewal[HR_ROR] * model.ror_renewal_binary[PD, HR_ROR] for HR_ROR in hr_ror) \
                       + sum(
                capacity_day_renewal[HR_DAY] * model.day_renewal_binary[PD, HR_DAY] for HR_DAY in hr_day) \
                       + sum(
                capacity_month_renewal[HR_MO] * model.month_renewal_binary[PD, HR_MO] for HR_MO in hr_mo)
    if pump_continous:
        cap_val += sum(model.capacity_storage[PD, ABA] for ABA in aba)

    return cap_val >= peak_demand[PD + '.' + SEAS] * (reserve_margin + 1)

model.planning_reserve = Constraint(pds, season, rule=planning_reserve)



###constrain retirements to extant plants
def retire(model, PD, ABA, TP):
    ind = pds.index(PD)
    return model.retire_therm[PD, ABA, TP] <= extant_thermal[pds[0] + '.' + ABA + '.' + TP] + sum(
        model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind])


model.retire = Constraint(pds, aba, tplants, rule=retire)


#### forces the model to retire the plants that their lifes ended
def lifetime(model, PD, ABA, TP):
    ind = pds.index(PD)
    return extant_thermal[pds[0] + '.' + ABA + '.' + TP] - sum(
        model.retire_therm[PDD, ABA, TP] for PDD in pds[:ind + 1]) <= extant_thermal[PD + '.' + ABA + '.' + TP]


model.lifetime = Constraint(pds, aba, tplants, rule=lifetime)


###wind generation
def windg(model, PD, H, ABA):
    ind = pds.index(PD)
    return model.windout[PD, H, ABA] == sum(
        (model.capacity_wind[PPD, GL] + model.capacity_wind_recon[PPD, GL]) * windcf[str(H) + '.' + str(GL)] for PPD in
        pds[:ind + 1] for GL in gl if ABA == map_gl_to_ba[int(GL)]) + extant_wind_gen[PD + '.' + ABA + '.' + str(H)]


model.windg = Constraint(pds, h, aba, rule=windg)


####solar generation
def solarg(model, PD, H, ABA):
    ind = pds.index(PD)
    return model.solarout[PD, H, ABA] == sum(
        (model.capacity_solar[PPD, GL] + model.capacity_solar_recon[PPD, GL]) * solarcf[str(H) + '.' + str(GL)] for PPD
        in pds[:ind + 1] for GL in gl if ABA == map_gl_to_ba[int(GL)]) + extant_solar_gen[PD + '.' + ABA + '.' + str(H)]


model.solarg = Constraint(pds, h, aba, rule=solarg)


def wind_recon(model, PD, GL):
    ind = pds.index(PD)
    return model.capacity_wind_recon[PD, GL] <= wind_recon_capacity[PD + '.' + GL] + sum(
        wind_recon_capacity[PPD + '.' + GL] - model.capacity_wind_recon[PPD, GL] for PPD in pds[:ind])


model.wind_recon = Constraint(pds, gl, rule=wind_recon)


def solar_recon(model, PD, GL):
    ind = pds.index(PD)
    return model.capacity_solar_recon[PD, GL] <= solar_recon_capacity[PD + '.' + GL] + sum(
        solar_recon_capacity[PPD + '.' + GL] - model.capacity_solar_recon[PPD, GL] for PPD in pds[:ind])


model.solar_recon = Constraint(pds, gl, rule=solar_recon)

##provincial supply and demand balance
if autrarky:
    def autrarky(model, PD, AP):
        TP_supply = sum(model.supply[PD, H, ABA, TP] for H in h for ABA in aba for TP in tplants if AP in ABA)
        wind_solar_supply = sum(model.windout[PD, H, ABA] for H in h for ABA in aba if AP in ABA) \
                            + sum(model.solarout[PD, H, ABA] for H in h for ABA in aba if AP in ABA)
        hydro_supply = sum(ror_hydroout[PD + '.' + str(H) + '.' + ABA] for H in h for ABA in aba if AP in ABA) \
                       + sum(model.daystoragehydroout[PD, H, ABA] for H in h for ABA in aba if AP in ABA) \
                       + sum(model.monthstoragehydroout[PD, H, ABA] for H in h for ABA in aba if AP in ABA)
        return TP_supply + wind_solar_supply + hydro_supply >= autonomy_pct * sum(
            demand[PD + '.' + ABA + '.' + str(H)] for ABA in aba for H in h if AP in ABA)


    model.autrarky = Constraint(pds, ap, rule=autrarky)

###supply and demand balance
aux = [1]


def demsup(model, PD, H, ABA):
    TP_supply = sum(model.supply[PD, H, ABA, TP] for TP in tplants)
    wind_solar_supply = model.windout[PD, H, ABA] + model.solarout[PD, H, ABA]
    hydro_supply = ror_hydroout[PD + '.' + str(H) + '.' + ABA] + model.daystoragehydroout[PD, H, ABA] + \
                   model.monthstoragehydroout[PD, H, ABA] + model.pumpout[PD, H, ABA] - model.pumpin[PD, H, ABA]
    renewal_supply = 0
    if hydro_development:
        renewal_supply = sum(
            ror_renewalout[str(H) + '.' + HR_ROR] * model.ror_renewal_binary[PD, HR_ROR] for HR_ROR in hr_ror if
            ABA == hr_ror_location[HR_ROR]) \
                         + sum(
            model.dayrenewalout[PD, H, HR_DAY] for HR_DAY in hr_day if ABA == hr_day_location[HR_DAY]) \
                         + sum(
            model.monthrenewalout[PD, H, HR_MO] for HR_MO in hr_mo if ABA == hr_month_location[HR_MO])
    return TP_supply + wind_solar_supply + hydro_supply + renewal_supply >= demand[PD + '.' + ABA + '.' + str(H)] + sum(
        demand_us[ABA + '.' + str(H)] for i in aux if ABA + '.' + str(H) in demand_us) \
           + sum(
        model.transmission[PD, H, ABA, ABBA] - (1 - transloss[ABBA + '.' + ABA]) * model.transmission[PD, H, ABBA, ABA]
        for ABBA in aba if ABA + '.' + ABBA in transmap)


model.demsup = Constraint(pds, h, aba, rule=demsup)


###maximum annual capacity factor for thermal plants
def maxcapfactor(model, PD, ABA, TP):
    ind = pds.index(PD)
    return sum(model.supply[PD, H, ABA, TP] for H in h) <= (
                sum(model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1]) +
                extant_thermal[pds[0] + '.' + ABA + '.' + TP]) * hours * max_cap_fact[TP]


model.maxcapfactor = Constraint(pds, aba, tplants, rule=maxcapfactor)


###minimum annual capacity factor for thermal plants

def mincapfactor(model, PD, ABA, TP):
    ind = pds.index(PD)
    return sum(model.supply[PD, H, ABA, TP] for H in h) >= (
                sum(model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1]) +
                extant_thermal[pds[0] + '.' + ABA + '.' + TP]) * hours * min_cap_fact[TP]


model.mincapfactor = Constraint(pds, aba, tplants, rule=mincapfactor)


##transmission capacity constraint
def transcap(model, PD, H, ABA, ABBA):
    ind = pds.index(PD)
    return model.transmission[PD, H, ABA, ABBA] <= sum(
        model.capacity_transmission[PPD, ABA, ABBA] for PPD in pds[:ind + 1] for i in aux if
        ABA + '.' + ABBA in transmap) + sum(
        extant_transmission[ABA + '.' + ABBA] for i in aux if ABA + '.' + ABBA in extant_transmission)


model.transcap = Constraint(pds, h, aba, abba, rule=transcap)


#####capacity constraints for thermal plants
def cap(model, PD, H, ABA, TP):
    ind = pds.index(PD)
    return model.supply[PD, H, ABA, TP] <= sum(
        model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1]) + \
           extant_thermal[pds[0] + '.' + ABA + '.' + TP]


model.cap = Constraint(pds, h, aba, tplants, rule=cap)

### this constraint limits the PHS retrofit capacity to percentage of available hydro reservior facility in each BA
if pump_continous:
    def pumpretrofitlimit(model, PD, ABA):
        ind = pds.index(PD)
        return sum(model.capacity_storage[PPD, ABA] for PPD in pds[:ind + 1]) <= (
                    day_hydro_capacity[pds[0] + '.' + ABA] + month_hydro_capacity[pds[0] + '.' + ABA]) * pump_ret_limit


    model.pumpretrofitlimit = Constraint(pds, aba, rule=pumpretrofitlimit)


#####pumped hydro energy storage
def pumpen(model, PD, H, ABA):
    return model.pumpenergy[PD, H + time_diff[H], ABA] == model.pumpenergy[PD, H, ABA] - model.pumpout[PD, H, ABA] + \
           model.pumpin[PD, H, ABA] * pump_hydro_efficiency


model.pumpen = Constraint(pds, h2, aba, rule=pumpen)


######pumped hydro energy storage
def pumcap(model, PD, H, ABA):
    ind = pds.index(PD)
    pump_new_con = 0
    pump_integer_cap = 0

    if pump_continous:
        pump_new_con = model.capacity_storage[PD, ABA]
    if hydro_development and not pump_continous:
        pump_integer_cap = sum(
            model.pumphydro[PPD, HR_PUMP] * capacity_pump_renewal[HR_PUMP] for PPD in pds[:ind + 1] for HR_PUMP in
            hr_pump if hr_pump_location[HR_PUMP] == ABA)
    return model.pumpenergy[PD, H, ABA] <= (ba_pump_hydro_capacity[ABA] + pump_integer_cap + pump_new_con) * pump_hours


model.pumcap = Constraint(pds, h, aba, rule=pumcap)


######pump hydro power capacity
def pumpoutmax(model, PD, H, ABA):
    ind = pds.index(PD)
    pump_new_con = 0
    pump_integer_cap = 0

    if pump_continous:
        pump_new_con = model.capacity_storage[PD, ABA]
    if hydro_development and not pump_continous:
        pump_integer_cap = sum(
            model.pumphydro[PPD, HR_PUMP] * capacity_pump_renewal[HR_PUMP] for PPD in pds[:ind + 1] for HR_PUMP in
            hr_pump if hr_pump_location[HR_PUMP] == ABA)

    return model.pumpout[PD, H, ABA] <= ba_pump_hydro_capacity[ABA] + pump_integer_cap + pump_new_con


model.pumpoutmax = Constraint(pds, h, aba, rule=pumpoutmax)


#####pump hydro pumping capacity
def pumpinmax(model, PD, H, ABA):
    ind = pds.index(PD)

    pump_new_con = 0
    pump_integer_cap = 0

    if pump_continous:
        pump_new_con = model.capacity_storage[PD, ABA]
    if hydro_development and not pump_continous:
        pump_integer_cap = sum(
            model.pumphydro[PPD, HR_PUMP] * capacity_pump_renewal[HR_PUMP] for PPD in pds[:ind + 1] for HR_PUMP in
            hr_pump if hr_pump_location[HR_PUMP] == ABA)

    return model.pumpin[PD, H, ABA] * pump_hydro_efficiency <= ba_pump_hydro_capacity[
        ABA] + pump_integer_cap + pump_new_con


model.pumpinmax = Constraint(pds, h, aba, rule=pumpinmax)


#####hydro storage for systems with intra-day storage
def hydro_daystorage(model, PD, D, ABA):
    return sum(model.daystoragehydroout[PD, H, ABA] for H in h if map_hd[H] == D) <= day_hydro_historic[
        PD + '.' + str(D) + '.' + ABA]


model.hydro_daystorage = Constraint(pds, d, aba, rule=hydro_daystorage)

if hydro_development:
    def hydro_dayrenewal(model, PD, D, HR_DAY):
        return sum(model.dayrenewalout[PD, H, HR_DAY] for H in h if map_hd[H] == D) <= day_renewal_historic[
            str(D) + '.' + HR_DAY]


    model.hydro_dayrenewal = Constraint(pds, d, hr_day, rule=hydro_dayrenewal)


###hydro storage for systems with intra-month storage
def hydro_monthstorage(model, PD, M, ABA):
    return sum(model.monthstoragehydroout[PD, H, ABA] for H in h if map_hm[H] == M) <= month_hydro_historic[
        PD + '.' + str(M) + '.' + ABA]


model.hydro_monthstorage = Constraint(pds, m, aba, rule=hydro_monthstorage)

if hydro_development:
    def hydro_monthrenewal(model, PD, M, HR_MO):
        return sum(model.monthrenewalout[PD, H, HR_MO] for H in h if map_hm[H] == M) <= month_renewal_historic[
            str(M) + '.' + HR_MO]


    model.hydro_monthrenewal = Constraint(pds, m, hr_mo, rule=hydro_monthrenewal)


####hydro minimum flow constraints for systems with intra-day storage
def hydro_dayminflow(model, PD, H, ABA):
    return model.daystoragehydroout[PD, H, ABA] >= day_minflow[PD + '.' + ABA]


model.hydro_dayminflow = Constraint(pds, h, aba, rule=hydro_dayminflow)

if hydro_development:
    def renewal_dayminflow(model, PD, H, HR_DAY):
        return model.dayrenewalout[PD, H, HR_DAY] >= capacity_day_renewal[HR_DAY] * hydro_minflow * \
               model.day_renewal_binary[PD, HR_DAY]


    model.renewal_dayminflow = Constraint(pds, h, hr_day, rule=renewal_dayminflow)


####hydro minimum flow constraints for systems with intra-month storage
def hydro_monthminflow(model, PD, H, ABA):
    return model.monthstoragehydroout[PD, H, ABA] >= month_minflow[PD + '.' + ABA]


model.hydro_monthminflow = Constraint(pds, h, aba, rule=hydro_monthminflow)

if hydro_development:
    def renewal_monthminflow(model, PD, H, HR_MO):
        return model.monthrenewalout[PD, H, HR_MO] >= capacity_month_renewal[HR_MO] * hydro_minflow * \
               model.month_renewal_binary[PD, HR_MO]


    model.renewal_monthminflow = Constraint(pds, h, hr_mo, rule=renewal_monthminflow)


##hydro capacity constraints for systems with intra-day storage
def hydro_daycap(model, PD, H, ABA):
    return model.daystoragehydroout[PD, H, ABA] <= day_hydro_capacity[PD + '.' + ABA]


model.hydro_daycap = Constraint(pds, h, aba, rule=hydro_daycap)

if hydro_development:
    def renewal_daycap(model, PD, H, HR_DAY):
        return model.dayrenewalout[PD, H, HR_DAY] <= capacity_day_renewal[HR_DAY] * model.day_renewal_binary[PD, HR_DAY]


    model.renewal_daycap = Constraint(pds, h, hr_day, rule=renewal_daycap)


#####hydro capacity constraints for systems with intra-month storage
def hydro_monthcap(model, PD, H, ABA):
    return model.monthstoragehydroout[PD, H, ABA] <= month_hydro_capacity[PD + '.' + ABA]


model.hydro_monthcap = Constraint(pds, h, aba, rule=hydro_monthcap)

if hydro_development:
    def renewal_monthcap(model, PD, H, HR_MO):
        return model.monthrenewalout[PD, H, HR_MO] <= capacity_month_renewal[HR_MO] * model.month_renewal_binary[
            PD, HR_MO]


    model.renewal_monthcap = Constraint(pds, h, hr_mo, rule=renewal_monthcap)

##### The following constraints ensure that the model does not build a hydro renewal or greenfield project more that one time during all periods (pds)
if hydro_development:
    def ror_onetime(model, HR_ROR):
        return sum(model.ror_renewal_binary[PD, HR_ROR] for PD in pds) <= 1


    model.ror_onetime = Constraint(hr_ror, rule=ror_onetime)


    def day_onetime(model, HR_DAY):
        return sum(model.day_renewal_binary[PD, HR_DAY] for PD in pds) <= 1


    model.day_onetime = Constraint(hr_day, rule=day_onetime)


    def month_onetime(model, HR_MO):
        return sum(model.month_renewal_binary[PD, HR_MO] for PD in pds) <= 1


    model.month_onetime = Constraint(hr_mo, rule=month_onetime)


# up ramp limit
def ramp_up(model, PD, H, ABA, TP):
    ind = pds.index(PD)
    return model.supply[PD, H + time_diff[H], ABA, TP] <= model.supply[PD, H, ABA, TP] + (
                sum(model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1]) +
                extant_thermal[pds[0] + '.' + ABA + '.' + TP]) * ramp_rate_percent[TP] * time_diff[H]


model.ramp_up = Constraint(pds, h2, aba, tplants, rule=ramp_up)


# down ramp limit
def ramp_down(model, PD, H, ABA, TP):
    ind = pds.index(PD)
    return model.supply[PD, H + time_diff[H], ABA, TP] >= model.supply[PD, H, ABA, TP] - (
                sum(model.capacity_therm[PPD, ABA, TP] - model.retire_therm[PPD, ABA, TP] for PPD in pds[:ind + 1]) +
                extant_thermal[pds[0] + '.' + ABA + '.' + TP]) * ramp_rate_percent[TP] * time_diff[H]


model.ramp_down = Constraint(pds, h2, aba, tplants, rule=ramp_down)


# capacity limit for wind plants
def windcaplimit(model, GL):
    return sum(model.capacity_wind[PD, GL] + model.capacity_wind_recon[PD, GL] for PD in pds) <= maxwind[GL]


model.windcaplimit = Constraint(gl, rule=windcaplimit)


# capacity limit for solar plants
def solarcaplimit(model, GL):
    return sum(model.capacity_solar[PD, GL] + model.capacity_solar_recon[PD, GL] for PD in pds) <= maxsolar[GL]


model.solarcaplimit = Constraint(gl, rule=solarcaplimit)

###################carbon limit constraint can be on or off##################
if emission_limit:
    def carbonlimit(model, AP):
        return sum(
            model.supply[pds[-1], H, ABA, TP] * carbondioxide[TP] / 1000000 for H in h for TP in tplants for ABA in aba
            if AP in ABA) <= carbon_limit[AP] / (365 / len(rundays))


    model.carbonlimit = Constraint(ap, rule=carbonlimit)

if min_installed_gas_PHP:
    def installedgasPHP(model, ABA):
        return sum(
            model.capacity_therm[PD, ABA, 'peaker'] - model.retire_therm[PD, ABA, 'peaker'] + model.capacity_storage[
                PD, ABA] for PD in pds) + extant_thermal[pds[0] + '.' + ABA + '.' + 'peaker'] + ba_pump_hydro_capacity[
                   ABA] >= gas_PHP_installed_limit[ABA]


    model.installedgasPHP = Constraint(aba, rule=installedgasPHP)

end = time.time()

print(f'\n==================================================\n\
Creating model time: {round((end - start) / 60)} Min and {round((end - start) % 60)} Sec \
\n==================================================')

start = time.time()
# solve the LP
opt = SolverFactory('cplex')
# opt.options['absmipgap'] = 0.05
# opt.options['optimality'] = 0.01
result_obj = opt.solve(model, tee=True)
result_obj.write()

end = time.time()

print(f'\n==================================================\n\
Solving process time: {round((end - start) / 60)} Min and {round((end - start) % 60)} Sec\
\n==================================================')

# Analyze the results
# production=dict()
# totdemand=dict()
# supminusdem=dict()
# carbonvalue=dict()
# thermal_production=dict()
# for AP in ap:
#     carbonvalue[AP]=sum(model.supply[H,ABA,TP].value*carbondioxide[TP]/1000000 for H in h for TP in tplants for ABA in aba if AP in ABA)
# for ABA in aba:
#     for H in h:
#         thermal_production[ABA+'.'+str(H)]=sum(model.supply[H,ABA,TP].value for TP in tplants)
#         production[ABA+'.'+str(H)]=thermal_production[ABA+'.'+str(H)]+model.windout[H,ABA].value+model.solarout[H,ABA].value+ror_hydroout[str(H)+'.'+ABA]+model.daystoragehydroout[H,ABA].value\
#                                           +model.monthstoragehydroout[H,ABA].value+model.pumpout[H,ABA].value-model.pumpin[H,ABA].value

#         totdemand[ABA+'.'+str(H)]=demand[ABA+'.'+str(H)]+sum(demand_us[ABA+'.'+str(H)] for i in aux if ABA+'.'+str(H) in demand_us) +sum(model.transmission[H,ABA,ABBA].value\
#                                         -(1-transloss[ABA+'.'+ABBA])*model.transmission[H,ABA,ABBA].value for ABBA in aba if ABA+'.'+ABBA in transmap)

#         supminusdem[H]=production[ABA+'.'+str(H)]-totdemand[ABA+'.'+str(H)]


############Saving resaults in .csv files################
folder_name = 'outputs' + '_ct' + str(ctax) + '_rd' + str(len(rundays)) + '_pds' + str(len(pds))

if test:
    folder_name += '_Test'
if hierarchical:
    folder_name += '_Hr'
elif downsampling:
    folder_name += '_DS'

if not OBPS_on:
    folder_name += '_NoOBPS'
else:
    folder_name += '_OBPS'
if local_gas_price:
    folder_name += '_LGP'
if not hydro_development:
    folder_name += '_NoHydro'
else:
    folder_name += '_Hydro'
if not emission_limit:
    folder_name += '_NoCL'
else:
    folder_name += '_CL' + str(int(carbon_reduction * 100))
if pump_continous:
    folder_name += '_CPHy'
else:
    folder_name += '_NoCPHy'
if not autrarky:
    folder_name += '_NoAr'
else:
    folder_name += '_Ar'
if SMR_CCS:
    folder_name += '_SMR_CCS'
else:
    folder_name += '_NoSMR_CCS'

if CPO:
    folder_name += '_CPO'

if GPS:
    folder_name += '_GPS'

cwd = '/project/def-mcpher16/ml_shared/COPPER5_CapCost_CTax_PopGrowth_CCS_True/outputs/'
os.makedirs(cwd, exist_ok=True)
os.chdir(cwd)
os.makedirs(folder_name, exist_ok=True)
outputdir = cwd + '/' + folder_name
os.chdir(outputdir)

# outputfilename='_dynamic_fy'+str(foryear)+'_ct'+str(ctax)+'_rd'+str(rundaynum)+'_sr'+str(sample_rate)

ind = list(model.capacity_therm)
val = list(model.capacity_therm[:, :, :].value)
capacity_thermal = [i + tuple([j]) for i, j in zip(ind, val)]
# retire_thermal = np.asarray(resultP)
np.savetxt('capacity_thermal.csv', capacity_thermal, fmt='%s', delimiter=',')

ind = list(model.retire_therm)
val = list(model.retire_therm[:, :, :].value)
retire_thermal = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('retire_thermal.csv', retire_thermal, fmt='%s', delimiter=',')

ind = list(model.capacity_wind)
val = list(model.capacity_wind[:, :].value)
capacity_wind = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('capacity_wind.csv', capacity_wind, fmt='%s', delimiter=',')

ind = list(model.capacity_solar)
val = list(model.capacity_solar[:, :].value)
capacity_solar = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('capacity_solar.csv', capacity_solar, fmt='%s', delimiter=',')

ind = list(model.supply)
val = list(model.supply[:, :, :, :].value)
supply = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('supply.csv', supply, fmt='%s', delimiter=',')

ind = list(model.windout)
val = list(model.windout[:, :, :].value)
windout = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('windout.csv', windout, fmt='%s', delimiter=',')

ind = list(model.solarout)
val = list(model.solarout[:, :, :].value)
solarout = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('solarout.csv', solarout, fmt='%s', delimiter=',')

ind = list(model.pumpout)
val = list(model.pumpout[:, :, :].value)
pumpout = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('pumpout.csv', pumpout, fmt='%s', delimiter=',')

ind = list(model.pumpin)
val = list(model.pumpin[:, :, :].value)
pumpin = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('pumpin.csv', pumpin, fmt='%s', delimiter=',')

ind = list(model.pumpenergy)
val = list(model.pumpenergy[:, :, :].value)
pumpenergy = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('pumpenergy.csv', pumpenergy, fmt='%s', delimiter=',')

ind = list(model.daystoragehydroout)
val = list(model.daystoragehydroout[:, :, :].value)
daystoragehydroout = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('daystoragehydroout.csv', daystoragehydroout, fmt='%s', delimiter=',')

ind = list(model.monthstoragehydroout)
val = list(model.monthstoragehydroout[:, :, :].value)
monthstoragehydroout = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('monthstoragehydroout.csv', monthstoragehydroout, fmt='%s', delimiter=',')

ind = list(model.transmission)
val = list(model.transmission[:, :, :, :].value)
transmission = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('transmission.csv', transmission, fmt='%s', delimiter=',')

ind = list(model.capacity_transmission)
val = list(model.capacity_transmission[:, :, :].value)
capacity_transmission = [i + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('capacity_transmission.csv', capacity_transmission, fmt='%s', delimiter=',')

if hydro_development:
    ind = list(model.ror_renewal_binary)
    val = list(model.ror_renewal_binary[:, :].value)
    ror_renewal_binary = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('ror_renewal_binary.csv', ror_renewal_binary, fmt='%s', delimiter=',')

    ind = list(model.day_renewal_binary)
    val = list(model.day_renewal_binary[:, :].value)
    day_renewal_binary = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('day_renewal_binary.csv', day_renewal_binary, fmt='%s', delimiter=',')

    ind = list(model.month_renewal_binary)
    val = list(model.month_renewal_binary[:, :].value)
    month_renewal_binary = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('month_renewal_binary.csv', month_renewal_binary, fmt='%s', delimiter=',')
    if not pump_continous:
        ind = list(model.pumphydro)
        val = list(model.pumphydro[:, :].value)
        pumphydro = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
        np.savetxt('pumphydro.csv', pumphydro, fmt='%s', delimiter=',')

    ind = list(model.dayrenewalout)
    val = list(model.dayrenewalout[:, :, :].value)
    dayrenewalout = [i + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('dayrenewalout.csv', dayrenewalout, fmt='%s', delimiter=',')

    ind = list(model.monthrenewalout)
    val = list(model.monthrenewalout[:, :, :].value)
    monthrenewalout = [i + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('monthrenewalout.csv', monthrenewalout, fmt='%s', delimiter=',')

if pump_continous:
    ind = list(model.capacity_storage)
    val = list(model.capacity_storage[:, :].value)
    capacity_storage = [i + tuple([j]) for i, j in zip(ind, val)]
    np.savetxt('capacity_storage.csv', capacity_storage, fmt='%s', delimiter=',')

ind = ['Objective_function_value']
val = list([model.obj()])
obj = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('obj_value.csv', obj, fmt='%s', delimiter=',')

ind = list(model.capacity_wind_recon)
val = list(model.capacity_wind_recon[:, :].value)
capacity_wind_recon = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('capacity_wind_recon.csv', capacity_wind_recon, fmt='%s', delimiter=',')

ind = list(model.capacity_solar_recon)
val = list(model.capacity_solar_recon[:, :].value)
capacity_solar_recon = [tuple([i]) + tuple([j]) for i, j in zip(ind, val)]
np.savetxt('capacity_solar_recon.csv', capacity_solar_recon, fmt='%s', delimiter=',')

number_run_days = len(rundays)
original_stdout = sys.stdout  # Save a reference to the original standard output
with open('COPPER config.txt', 'w') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print(f'Planning for the target year {foryear} considering {refyear} as the reference year')
    print(f'modeled {pds} palnning periods and ran {number_run_days} representative days in each period')
    print(f'Carbon price = {ctax}')
    print(f'reserve margin = {reserve_margin}')
    print(f'pumped hydro retrofit limit = {pump_ret_limit}')
    print(f'down sampling clustering ? {downsampling}')
    print(f'hierarchical clustering ? {hierarchical}')
    print(f'test run ? {test}')
    print(f'hydro development on ? {hydro_development}')
    print(f'autrarky on ? {autrarky}')
    print(f'pump as continous variable ? {pump_continous}')
    print(f'emission limit on?  {emission_limit}')
    print(f'local gas price on? {local_gas_price}')
    print(f'OBPS on? {OBPS_on}')
    print(f'SMR and CCS technologies ? {ctax}')
    print(f'emission limit reference year ? {emission_limit_ref_year}')
    print(f'Carbon reduction compared to refrence year {emission_limit_ref_year} = {carbon_reduction}')

    sys.stdout = original_stdout  # Reset the standard output to its original value

# model.ror_renewal_binary.pprint()
# model.day_renewal_binary.pprint()
# model.month_renewal_binary.pprint()
# model.pumphydro.pprint()

os.chdir(cwd)

############# Analyze the results ##################

#os.chdir('C://Users/SESIT Group/Desktop/Google Drive/PhD/Thesis/COPPER Paper/Results/OBS-GSP-CPO')

os.chdir(outputdir)

capacity_thermal = pd.read_csv(r'capacity_thermal.csv', header=None)

capacity_storage = pd.read_csv(r'capacity_storage.csv', header=None)

retire_thermal = pd.read_csv(r'retire_thermal.csv', header=None)

capacity_wind = pd.read_csv(r'capacity_wind.csv', header=None)

capacity_solar = pd.read_csv(r'capacity_solar.csv', header=None)

supply = pd.read_csv(r'supply.csv', header=None)

windout = pd.read_csv(r'windout.csv', header=None)

solarout = pd.read_csv(r'solarout.csv', header=None)

pumpout = pd.read_csv(r'pumpout.csv', header=None)

pumpin = pd.read_csv(r'pumpin.csv', header=None)

pumpenergy = pd.read_csv(r'pumpenergy.csv', header=None)

daystoragehydroout = pd.read_csv(r'daystoragehydroout.csv', header=None)

monthstoragehydroout = pd.read_csv(r'monthstoragehydroout.csv', header=None)

transmission = pd.read_csv(r'transmission.csv', header=None)

capacity_transmission = pd.read_csv(r'capacity_transmission.csv', header=None)
if hydro_development:
    ror_renewal_binary = pd.read_csv(r'ror_renewal_binary.csv', header=None)

    day_renewal_binary = pd.read_csv(r'day_renewal_binary.csv', header=None)

    month_renewal_binary = pd.read_csv(r'month_renewal_binary.csv', header=None)

    #    pumphydro = pd.read_csv(r'pumphydro.csv', header=None)

    dayrenewalout = pd.read_csv(r'dayrenewalout.csv', header=None)

    monthrenewalout = pd.read_csv(r'monthrenewalout.csv', header=None)

obj = pd.read_csv(r'obj_value.csv', header=None)

capacity_wind_recon = pd.read_csv(r'capacity_wind_recon.csv', header=None)
capacity_solar_recon = pd.read_csv(r'capacity_solar_recon.csv', header=None)

######### Whole country Generation outline ##########
tp_num = len(tplants)
Canada_gen_outline = np.zeros((len(pds), len(allplants)))
capcitytherm = list(capacity_thermal.iloc[:, 3])
retiretherm = list(retire_thermal.iloc[:, 3])
Total_installed = np.zeros((len(pds), len(allplants)))
Total_retired = np.zeros((len(pds), len(tplants)))
Total_installed_hydro_aba = dict()
# Total_recon_installed=dict()
Total_generation_ABA = np.zeros((len(pds) * len(aba), len(allplants)))
Total_installed_ABA = np.zeros((len(aba), len(allplants)))
for PD in pds:
    for ALP in allplants:

        for ABA in aba:
            index_aba = aba.index(ABA)

            index_p = allplants.index(ALP)
            index_pd = pds.index(PD)

            if ALP != 'wind' and ALP != 'solar' and ALP != 'hydro':
                index_tp = tplants.index(ALP)
                Canada_gen_outline[index_pd, index_p] += extant_thermal[pds[0] + '.' + ABA + '.' + ALP] + sum(
                    capcitytherm[ii * len(tplants) * len(aba) + index_aba * tp_num + index_tp] - retiretherm[
                        ii * len(tplants) * len(aba) + index_aba * tp_num + index_tp] for ii in range(index_pd + 1))
                Total_installed[index_pd, index_p] += capcitytherm[
                    index_pd * len(tplants) * len(aba) + index_aba * tp_num + index_tp]
                Total_retired[index_pd, index_tp] += retiretherm[
                    index_pd * len(tplants) * len(aba) + index_aba * tp_num + index_tp]
                Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += extant_thermal[pds[
                                                                                                     0] + '.' + ABA + '.' + ALP] + sum(
                    capcitytherm[ii * len(tplants) * len(aba) + index_aba * tp_num + index_tp] - retiretherm[
                        ii * len(tplants) * len(aba) + index_aba * tp_num + index_tp] for ii in range(index_pd + 1))
                Total_installed_ABA[index_aba, index_p] += capcitytherm[
                    index_pd * len(tplants) * len(aba) + index_aba * tp_num + index_tp]
            elif ALP == 'hydro':

                Canada_gen_outline[index_pd, index_p] += ror_hydro_capacity[PD + '.' + ABA] + day_hydro_capacity[
                    PD + '.' + ABA] + month_hydro_capacity[PD + '.' + ABA]
                Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += ror_hydro_capacity[PD + '.' + ABA] + \
                                                                                  day_hydro_capacity[PD + '.' + ABA] + \
                                                                                  month_hydro_capacity[PD + '.' + ABA]
                if hydro_development:
                    Total_installed_hydro_aba[PD + '.' + ABA + '.ror'] = 0
                    Total_installed_hydro_aba[PD + '.' + ABA + '.day'] = 0
                    Total_installed_hydro_aba[PD + '.' + ABA + '.month'] = 0

                    for HR_ROR in hr_ror:
                        if ABA == location_renewal[HR_ROR]:
                            index_rn = hr_ror.index(HR_ROR)
                            Canada_gen_outline[index_pd, index_p] += sum(
                                ror_renewal_binary.iloc[index_rn + ii * len(hr_ror)][2] * capacity_renewal[HR_ROR] for
                                ii in range(index_pd + 1))
                            Total_installed[index_pd, index_p] += \
                            ror_renewal_binary.iloc[index_rn + index_pd * len(hr_ror)][2] * capacity_renewal[HR_ROR]
                            Total_installed_hydro_aba[PD + '.' + ABA + '.ror'] += \
                            ror_renewal_binary.iloc[index_rn + index_pd * len(hr_ror)][2] * capacity_renewal[HR_ROR]
                            Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += sum(
                                ror_renewal_binary.iloc[index_rn + ii * len(hr_ror)][2] * capacity_renewal[HR_ROR] for
                                ii in range(index_pd + 1))
                            Total_installed_ABA[index_aba, index_p] += \
                            ror_renewal_binary.iloc[index_rn + index_pd * len(hr_ror)][2] * capacity_renewal[HR_ROR]

                    for HR_DAY in hr_day:
                        if ABA == location_renewal[HR_DAY]:
                            index_rn = hr_day.index(HR_DAY)
                            Canada_gen_outline[index_pd, index_p] += sum(
                                day_renewal_binary.iloc[index_rn + ii * len(hr_day)][2] * capacity_renewal[HR_DAY] for
                                ii in range(index_pd + 1))
                            Total_installed[index_pd, index_p] += \
                            day_renewal_binary.iloc[index_rn + index_pd * len(hr_day)][2] * capacity_renewal[HR_DAY]
                            Total_installed_hydro_aba[PD + '.' + ABA + '.day'] += \
                            day_renewal_binary.iloc[index_rn + index_pd * len(hr_day)][2] * capacity_renewal[HR_DAY]
                            Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += sum(
                                day_renewal_binary.iloc[index_rn + ii * len(hr_day)][2] * capacity_renewal[HR_DAY] for
                                ii in range(index_pd + 1))
                            Total_installed_ABA[index_aba, index_p] += \
                            day_renewal_binary.iloc[index_rn + index_pd * len(hr_day)][2] * capacity_renewal[HR_DAY]

                    for HR_MO in hr_mo:
                        if ABA == location_renewal[HR_MO]:
                            index_rn = hr_mo.index(HR_MO)
                            Canada_gen_outline[index_pd, index_p] += sum(
                                month_renewal_binary.iloc[index_rn + ii * len(hr_mo)][2] * capacity_renewal[HR_MO] for
                                ii in range(index_pd + 1))
                            Total_installed[index_pd, index_p] += \
                            month_renewal_binary.iloc[index_rn + index_pd * len(hr_mo)][2] * capacity_renewal[HR_MO]
                            Total_installed_hydro_aba[PD + '.' + ABA + '.month'] += \
                            month_renewal_binary.iloc[index_rn + index_pd * len(hr_mo)][2] * capacity_renewal[HR_MO]
                            Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += sum(
                                month_renewal_binary.iloc[index_rn + ii * len(hr_mo)][2] * capacity_renewal[HR_MO] for
                                ii in range(index_pd + 1))
                            Total_installed_ABA[index_aba, index_p] += \
                            month_renewal_binary.iloc[index_rn + index_pd * len(hr_mo)][2] * capacity_renewal[HR_MO]



            elif ALP == 'wind' or ALP == 'solar':

                for GL in gl:
                    if map_gl_to_ba[int(GL)] == ABA and str(GL) + '.' + ALP in extant_wind_solar:
                        Canada_gen_outline[index_pd, index_p] += extant_wind_solar[str(GL) + '.' + ALP]
                        Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += extant_wind_solar[
                            str(GL) + '.' + ALP]
                    if map_gl_to_ba[int(GL)] == ABA and ALP == 'wind':
                        Canada_gen_outline[index_pd, index_p] += sum(capacity_wind.iloc[len(gl) * ii + int(GL) - 1][2] +
                                                                     capacity_wind_recon.iloc[
                                                                         len(gl) * ii + int(GL) - 1][2] for ii in
                                                                     range(index_pd + 1))
                        Total_installed[index_pd, index_p] += capacity_wind.iloc[len(gl) * index_pd + int(GL) - 1][2] + \
                                                              capacity_wind_recon.iloc[
                                                                  len(gl) * index_pd + int(GL) - 1][2]
                        Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += \
                        capacity_wind.iloc[len(gl) * index_pd + int(GL) - 1][2] + \
                        capacity_wind_recon.iloc[len(gl) * index_pd + int(GL) - 1][2]
                        Total_installed_ABA[index_aba, index_p] += capacity_wind.iloc[len(gl) * index_pd + int(GL) - 1][
                                                                       2] + capacity_wind_recon.iloc[
                                                                       len(gl) * index_pd + int(GL) - 1][2]

                    if map_gl_to_ba[int(GL)] == ABA and ALP == 'solar':
                        Canada_gen_outline[index_pd, index_p] += sum(
                            capacity_solar.iloc[len(gl) * ii + int(GL) - 1][2] +
                            capacity_solar_recon.iloc[len(gl) * ii + int(GL) - 1][2] for ii in range(index_pd + 1))
                        Total_installed[index_pd, index_p] += capacity_solar.iloc[len(gl) * index_pd + int(GL) - 1][2] + \
                                                              capacity_solar_recon.iloc[
                                                                  len(gl) * index_pd + int(GL) - 1][2]
                        Total_generation_ABA[len(aba) * index_pd + index_aba, index_p] += \
                        capacity_solar.iloc[len(gl) * index_pd + int(GL) - 1][2] + \
                        capacity_solar_recon.iloc[len(gl) * index_pd + int(GL) - 1][2]
                        Total_installed_ABA[index_aba, index_p] += \
                        capacity_solar.iloc[len(gl) * index_pd + int(GL) - 1][2] + \
                        capacity_solar_recon.iloc[len(gl) * index_pd + int(GL) - 1][2]

##### installed transmission ##########3
Installed_transmission = np.zeros((len(pds), len(transmap)))
tr_list = list(capacity_transmission.iloc[:][1])
iter_index = -1
for PD in pds:

    for ABA in aba:
        for ABBA in aba:
            if ABA + '.' + ABBA in transmap:
                Installed_transmission[pds.index(PD), transmap.index(ABA + '.' + ABBA)] = capacity_transmission.iloc[
                    pds.index(PD) * len(aba) * len(aba) + aba.index(ABA) * len(aba) + aba.index(ABBA)][3]

###### Carbon Emission by BA ######

carbon_ABA = np.zeros((len(pds), len(aba)))
hours_list = list(supply.iloc[:][0])
ba_list = list(supply.iloc[:][1])
tp_type = list(supply.iloc[:][2])
prod_power = list(supply.iloc[:][3])
carbon_national = dict()
for PD in pds:
    carbon_national[PD] = 0
    for H in h:
        for ABA in aba:
            for TP in tplants:
                carbon_ABA[pds.index(PD), aba.index(ABA)] += supply.iloc[pds.index(PD) * len(h) * len(aba) * len(
                    tplants) + h.index(H) * len(aba) * len(tplants) + aba.index(ABA) * len(tplants) + tplants.index(
                    TP)][4] * carbondioxide[TP] * 365 / len(run_days) / 1000000
                carbon_national[PD] += supply.iloc[pds.index(PD) * len(h) * len(aba) * len(tplants) + h.index(H) * len(
                    aba) * len(tplants) + aba.index(ABA) * len(tplants) + tplants.index(TP)][4] * carbondioxide[
                                           TP] * 365 / len(run_days) / 1000000

carbon_ap = np.zeros((len(pds), len(ap)))
for AP in ap:
    for PD in pds:
        for ABA in aba:
            if AP in ABA:
                carbon_ap[pds.index(PD), ap.index(AP)] += carbon_ABA[pds.index(PD), aba.index(ABA)]

Obj = obj.iloc[0][1] * 365 / len(run_days) / 1000000

carbon_ap = carbon_ap.transpose()
carbon_ABA = carbon_ABA.transpose()
Canada_gen_outline = Canada_gen_outline.transpose()
Total_installed = Total_installed.transpose()
Total_generation_ABA = Total_generation_ABA.transpose()

#carbon_ap.to_excel('carbon_ap', index=True)
CARBON_AP = pd.DataFrame(carbon_ap, columns = pds, index = ap)
CARBON_AP.to_excel('carbon_ap.xlsx', index=True)
#np.savetxt('carbon_ap.csv', carbon_ap, header='2030, 2040, 2050', fmt='%s', delimiter=",", comments="")
#CARBON_AP = pd.read_csv('carbon_ap.csv', header=True)
#CARBON_AP.insert(0, 'Provinces', ['British Columbia', "Alberta","Saskatchewan", "Manitoba", "Ontario","Quebec","New Brunswick","Newfoundland and Labrador","Nova Scotia","Prince Edward Island"], True)
#CARBON_AP.to_excel('carbon_ap.csv')

#carbon_ABA.to_excel('carbon_ABA', index=True)
#np.savetxt('carbon_ABA.csv', carbon_ABA, delimiter=",")
#Canada_gen_outline.to_excel('Canada_gen_outline', index=True)
#np.savetxt('canada_gen_outline.csv', Canada_gen_outline, header='2030, 2040, 2050', fmt='%s', delimiter=",", comments="")
CANADA_GEN_OUTLINE = pd.DataFrame(Canada_gen_outline, columns = pds, index = gendata.iloc[:]['Type'])
CANADA_GEN_OUTLINE.to_excel('Canada_gen_outline.xlsx', index=True)
#Total_installed.to_excel('Total_installed', index=True)
#np.savetxt('total_installed.csv', Total_installed, delimiter=",")
#Total_generation_ABA('Total_generation_ABA', index=True)
#np.savetxt('total_generation_ABA.csv', Total_generation_ABA, header='British Columbia.a, Alberta.a,  Saskatchewan.a, Manitoba.a, Ontario.a,Ontario.b,Quebec.a,Quebec.b, New Brunswick.a, Newfoundland and Labrador.a,Newfoundland and Labrador.b,Nova Scotia.a,Prince Edward Island.a,British Columbia.a, Alberta.a,  Saskatchewan.a, Manitoba.a, Ontario.a,Ontario.b,Quebec.a,Quebec.b, New Brunswick.a, Newfoundland and Labrador.a,Newfoundland and Labrador.b,Nova Scotia.a,Prince Edward Island.a,British Columbia.a, Alberta.a,  Saskatchewan.a, Manitoba.a, Ontario.a,Ontario.b,Quebec.a,Quebec.b, New Brunswick.a, Newfoundland and Labrador.a,Newfoundland and Labrador.b,Nova Scotia.a,Prince Edward Island.a', fmt='%s', delimiter=",")


TOTAL_generation_ABA_2030 = pd.DataFrame(Total_generation_ABA[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]], columns = aba, index = gendata.iloc[:]['Type'])
TOTAL_generation_ABA_2040 = pd.DataFrame(Total_generation_ABA[:, [13,14,15,16,17,18,19,20,21,22,23,24,25]], columns = aba, index = gendata.iloc[:]['Type'])
TOTAL_generation_ABA_2050 = pd.DataFrame(Total_generation_ABA[:, [26,27,28,29,30,31,32,33,34,35,36,37,38]], columns = aba, index = gendata.iloc[:]['Type'])
TOTAL_generation_ABA = pd.concat([TOTAL_generation_ABA_2030, TOTAL_generation_ABA_2040, TOTAL_generation_ABA_2050])
TOTAL_generation_ABA.to_excel('Total_Generation_ABA.xlsx', index=True)

TOTAL_generation_ap = TOTAL_generation_ABA
TOTAL_generation_ap['Ontario.a'] = TOTAL_generation_ABA['Ontario.a'] + TOTAL_generation_ABA['Ontario.b']
TOTAL_generation_ap['Quebec.a'] = TOTAL_generation_ABA['Quebec.a'] + TOTAL_generation_ABA['Quebec.b']
TOTAL_generation_ap['Newfoundland and Labrador.a'] = TOTAL_generation_ABA['Newfoundland and Labrador.a'] + TOTAL_generation_ABA['Newfoundland and Labrador.b']

TOTAL_generation_ap = TOTAL_generation_ap.drop(columns=['Ontario.b', 'Quebec.b', 'Newfoundland and Labrador.b'])
TOTAL_generation_ap.columns = ap

TOTAL_generation_ap.to_excel('Total_generation_ap.xlsx', index=True)



#model.ror_renewal_binary.pprint()
#model.day_renewal_binary.pprint()
#model.month_renewal_binary.pprint()
#model.pumphydro.pprint()



os.chdir(cwd)
