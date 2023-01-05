import os
import shutil

import pandas as pd
from numpy.random import uniform

"""
COPPER Input vars: 1) carbon tax 2) capital cost 3) demand growth (1~2) 4) technology toggles (nuclear on/off, 
hydro on/off, ccs on/off, transmission on/off)
two combinations of technology toggles: first 1) all on and 2) all off except transmission
"""

CCS = True

provinces_full = {
            "British Columbia": 4497392,
            "Alberta": 3584417,
            "Manitoba": 1234225,
            "New Brunswick": 717014,
            "Newfoundland and Labrador": 532691,
            "Nova Scotia": 944301,
            "Ontario": 12557108,
            "Quebec": 8190949,
            "Saskatchewan": 1078184,
            "Prince Edward Island": 140204
            }

pds=['2025','2030','2035','2040','2045','2050']



total_pop = sum(provinces_full.values())
pop_growth = []

path = os.getcwd()
os.makedirs(path+'/annual_growths',exist_ok=True)
os.makedirs(path+'/configurations',exist_ok=True)
os.makedirs(path+'/capital_costs',exist_ok=True)
os.makedirs(path+'/shells',exist_ok=True)
os.makedirs(path + '/scripts', exist_ok=True)
os.makedirs(path + '/ctax', exist_ok=True)

gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)
if CCS:
    gendata = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0)
    gendata_fix = pd.read_excel (r'Generation_type_data_SMR_CCS.xlsx',header=0, index_col=0)

else:
    gendata = pd.read_excel (r'Generation_type_data.xlsx',header=0)
    gendata_fix = pd.read_excel (r'Generation_type_data.xlsx',header=0, index_col=0)
    
capitalcost=dict(zip(list(gendata.iloc[:]['Type']),list(gendata.iloc[:]['capitalcost'])))
annualgrowth = pd.read_csv(r'annual_growth.csv', index_col=0)


configuration = pd.read_excel (r'COPPER_configuration.xlsx',header=0)
config=dict(zip(list(configuration.iloc[:]['Parameter']),list(configuration.iloc[:]['Value'])))
ctax=int(config['carbon price'])

max_th = 1.1
min_th = 0.9
max_vre = 1
min_vre = 0.6
#max_fp = 2
#min_fp = 0.9


for type in gendata.iloc[:]['Type']:
    locals()["r_cap_" + type] = uniform(0, 1, size=3000)

#for PDS in pds:
#    locals()["r_ctax_"+ PDS]= uniform(0.5, 2, size=3000)

r_ctax = uniform(0.5, 1.5, size=3000)
r_growth = uniform(0.1, 3.7, size=3000)

for i in range(0, 100):
    
    for type in gendata.iloc[:]['Type']:
        for thermal in gendata.iloc[:]['Is thermal?']:
            if thermal==True:
                gendata_fix.loc[type, 'capitalcost'] = capitalcost[type] * min_th + locals()["r_cap_" + type][i] * (
                        capitalcost[type] * max_th - capitalcost[type] * min_th)
            else:
                gendata_fix.loc[type, 'capitalcost'] = capitalcost[type] * min_vre + locals()["r_cap_" + type][i] * (
                        capitalcost[type] * max_vre - capitalcost[type] * min_vre)

    configuration.loc[4, 'Value'] = round(r_ctax[i]*430+70, 0)

    annualgrowth_new = annualgrowth.multiply(r_growth[i])
    provinces_new_pop = {}


    for province in provinces_full.keys():
        provinces_new_pop[province]= ((1 + annualgrowth_new.loc[province, '2030']) ** 12) * (
                (1 + annualgrowth_new.loc[province, '2040']) ** 10) * ((1 + annualgrowth_new.loc[province, '2050']
                                                                        ) ** 10)* provinces_full[province]

    pop_growth.append(sum(provinces_new_pop.values()) / total_pop)

    os.chdir(path+'/annual_growths')
    annualgrowth_new.to_csv('annual_growth_{}.csv'.format(i))
    os.chdir(path)
    os.chdir(path + '/configurations')
    configuration.to_excel('COPPER_configuration_{}.xlsx'.format(i),index=False)
    os.chdir(path)
    os.chdir(path + '/capital_costs')
    if CCS:
        gendata_fix.to_excel('Generation_type_data_SMR_CCS_{}.xlsx'.format(i),index=True)
    else:
        gendata_fix.to_excel('Generation_type_data_{}.xlsx'.format(i),index=True)
    
    os.chdir(path)


    shutil.copy2('COPPER5.1.py', 'scripts/COPPER5.1.py')
    os.chdir(path + '/scripts')
    file_name = 'COPPER5.1_{}.py'.format(i)
    # print(file_name)
    shutil.copy2('COPPER5.1.py', file_name)

    with open(file_name, 'r') as python_file:
        python_file_lines = python_file.readlines()

    python_file_lines[41] = "configuration = pd.read_excel (r'COPPER_configuration_" + str(
        i) + ".xlsx',header=0)" + '\n'
    python_file_lines[141] = "    gendata = pd.read_excel (r'Generation_type_data_SMR_CCS_" + str(
        i) + ".xlsx',header=0 )" + '\n'
    python_file_lines[143] = "    gendata = pd.read_excel (r'Generation_type_data_" + str(
        i) + ".xlsx',header=0 )" + '\n'
    python_file_lines[302] = "demand_growth = pd.read_csv(r'annual_growth_{}.csv',header=0,index_col=0)".format(
        i) + '\n'
    python_file_lines[1384] = "folder_name=str({}) + '_outputs'+'_ct'+str(ctax)+'_rd'+str(len(rundays))+" \
                              "'_pds'+str(len(pds))".format(i) + '\n'

    with open(file_name, 'w') as python_file:
        python_file.writelines(python_file_lines)
    
    os.chdir(path)

    shutil.copy2('COPPER5.sh', 'shells/COPPER5.sh')
    os.chdir(path + '/shells')
    file_name = 'COPPER5.1_' + str(i) + '.sh'
    shutil.copy2('COPPER5.sh', file_name)

    with open(file_name, 'rt') as shell_file:
        shell_file_lines = shell_file.readlines()
    

    shell_file_lines[2] = '#SBATCH --output=COPPER5_' + str(i) + '.out \n'
    shell_file_lines[8] = 'python /scratch/smmiri/COPPER5.1_' + str(i) + '.py \n'
    with open(file_name, 'w') as shell_file:
        shell_file.writelines(shell_file_lines)
    os.chdir(path)
    
pd.DataFrame(pop_growth).to_csv('pop_growth.csv', index=True, header=None)


