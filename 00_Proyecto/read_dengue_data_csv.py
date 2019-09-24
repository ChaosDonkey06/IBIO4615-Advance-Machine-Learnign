import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

dfs = pd.read_excel('./data/bases_dengue_semanaEpi.xls', sheet_name=None)
num_semanas_epi = 52
num_comunas = 13


cases = np.zeros((    344 , num_comunas ))

for year, sheet_name in enumerate(dfs.keys()):
    #print(year)
    cases_week_year = dfs[sheet_name]['SEMANA EPI']
    print(max(cases_week_year))
    cases_comuna_year = dfs[sheet_name]['comuna']
    for sem_epi in range(1,max(cases_week_year)+1):
        for comuna in range(1,num_comunas+1):
            semanas_epi_true = cases_week_year==sem_epi
            comuna_true = cases_comuna_year==comuna
            cases[(sem_epi-1)*(year+1),comuna-1] = np.sum(semanas_epi_true.mul(comuna_true))

np.savetxt('dengue_comuna_cases.csv', cases , delimiter=' ')


# 1 - 2,3,10,11,12
# 2 - 1,3
# 3 - 1,2,4,10
# 4 - 3,5,6,9,10
# 5 - 4,6,8,9

# 6 - 4,5,7,8
# 7 - 6,8
# 8 - 5,6,7,9
# 9 - 4,5,8,10
# 10 - 5,6,7,9

# 11 - 1,10,12
# 12 - 1,11,13
# 13 - 12

#      1 2 3 4 5 6 7 8 9 10 11 12 13
# 1  |[0 1 1 0 0 0 0 0 0 1  1  1  0 ]
# 2  |[0 0 1 0 0 0 0 0 0 0  0  0  0 ]
# 3  |[0 0 0 1 0 0 0 0 0 1  0  0  0 ]
# 4  |[0 0 0 0 1 1 0 0 1 1  0  0  0 ]
# 5  |[0 0 0 0 0 1 0 1 1 0  0  0  0 ]
# 6  |[0 0 0 0 0 0 1 1 0 0  0  0  0 ]
# 7  |[0 0 0 0 0 0 0 1 0 0  0  0  0 ]
# 8  |[0 0 0 0 0 0 0 0 1 0  0  0  0 ]
# 9  |[0 0 0 0 0 0 0 0 0 1  0  0  0 ]
# 10 |[0 0 0 0 0 0 0 0 0 0  0  0  0 ]
# 11 |[0 0 0 0 0 0 0 0 0 0  0  1  0 ]
# 12 |[0 0 0 0 0 0 0 0 0 0  0  0  1 ]
# 13 |[0 0 0 0 0 0 0 0 0 0  0  0  0 ]
com_rela=np.zeros((13,13))
com_rela[0,1]=1
com_rela[0,2]=1 
com_rela[0,9]=1
com_rela[0,10]=1
com_rela[0,11]=1
com_rela[1,2]=1
com_rela[2,3]=1
com_rela[2,9]=1
com_rela[3,4]=1
com_rela[3,5]=1
com_rela[3,8]=1
com_rela[3,9]=1

com_rela[4,5]=1
com_rela[4,7]=1
com_rela[4,8]=1

com_rela[5,6]=1
com_rela[5,7]=1
com_rela[6,7]=1
com_rela[7,8]=1
com_rela[8,9]=1
com_rela[10,11]=1
com_rela[11,12]=1

com_rela=com_rela+com_rela.transpose()

np.savetxt('./data/dengue_comuna_cases_relations.csv', com_rela , delimiter=' ')



cases = np.zeros((num_semanas_epi,num_comunas,len(dfs.keys())))
for year, sheet_name in enumerate(dfs.keys()):
    cases_week_year = dfs[sheet_name]['SEMANA EPI']
    cases_comuna_year = dfs[sheet_name]['comuna']
    for sem_epi in range(1,num_semanas_epi+1):
        for comuna in range(1,num_comunas+1):
            semanas_epi_true = cases_week_year==sem_epi
            comuna_true = cases_comuna_year==comuna


            cases[sem_epi-1,comuna-1,year] = np.sum(semanas_epi_true.mul(comuna_true))

cas_img = np.sum(cases,0)


fig, ax = plt.subplots()
fig2=plt.imshow(cas_img,cmap='OrRd')
cbar = fig.colorbar(fig2, ax=ax, extend='both')
cbar.minorticks_on()

ax.set_xlabel('Year'), ax.set_ylabel('Comuna')
years_=['','2013', '2015', '2017', '2019']
comunas_=['','1', '3', '5', '7',  '9', '11','13']
ax.set_xticklabels(years_)
ax.set_yticklabels(comunas_)