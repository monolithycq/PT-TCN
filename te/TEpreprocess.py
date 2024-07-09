import scipy.io as scio
import numpy as np
import pandas as pd
data=scio.loadmat('data_v2.mat')
print(type(data))
control=np.array(data['xmv'])
measure=np.array(data['simout'])
opCost=np.array(data['OpCost'])
control_del = np.delete(control,0,0)[::10,:]
measure_del = np.delete(measure,0,0)[::10,:]
opCost_del = np.delete(opCost,0,0)[::10,:]
all_data = np.hstack((measure_del,control_del))
data = pd.DataFrame(all_data)
writer = pd.ExcelWriter('data_v2.xlsx')		# 写入Excel文件
data.to_excel(writer, float_format='%.4f')
writer.save()

writer.close()
print(type(control_del))
print(type(measure))