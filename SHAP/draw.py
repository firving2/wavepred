# import pandas as pd
# import matplotlib.pyplot as plt
# # 从 CSV 文件中读取数据
# data = pd.read_csv('result_12.csv')
#
# # 提取四个类别的数据
# U10 = data['U10']
# V10 = data['V10']
# Mwp = data['Mwp']
# Swh = data['Swh']
#
# # 绘制箱型图
# plt.boxplot([U10, V10, Mwp, Swh], labels=['U10', 'V10', 'Mwp', 'Swh'])
#
# plt.ylabel('Mean |SHAP| ratios')
# plt.title('Boxplot of Variables')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 从 CSV 文件中读取数据
data = pd.read_csv('result_12.csv')

# 提取四个类别的数据
U10 = data['U10']
V10 = data['V10']
Mwp = data['Mwp']
Swh = data['Swh']

# 设置绘图风格
plt.style.use('science')

# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制箱型图
box_data = [U10, V10, Mwp, Swh]
ax.boxplot(box_data, labels=['U10', 'V10', 'Mwp', 'Swh'])

# 设置标签和标题
ax.set_ylabel('Mean |SHAP| ratios', fontsize=12)
ax.set_title('Boxplot of Variables', fontsize=14)

# 调整边距
plt.tight_layout()

# 显示图形
plt.show()