#使用pickle模块从文件中重构python对象

import pprint, pickle
graph_name='Surat'
pkl_file = open('./data/%s_coos.txt'%graph_name, 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

pkl_file.close()
