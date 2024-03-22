import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib
from scipy import interpolate
import scienceplots
# 数据
linear_regression_coefficients = [0.759, 0.816, 0.828, 0.751, 0.793]
gc_gru_coefficients = [0.997, 0.997, 0.997, 0.996, 0.997]
x = np.arange(len(linear_regression_coefficients))  # x轴坐标
width = 0.35  # 柱状图宽度
# with plt.style.context(['science', 'no-latex', 'cjk-sc-font']):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     matplotlib.rcParams.update({'font.size': 15})
#     # # 设置图形样式
#     # plt.style.use('seaborn-deep')  # 使用深色主题
#
#     # 绘制柱状图
#     rects1 = ax.bar(x - width / 2, linear_regression_coefficients, width, label='Linear Regression', color='lightgreen')
#     rects2 = ax.bar(x + width / 2, gc_gru_coefficients, width, label='GC-GRU model', color='blue')
#
#     # 添加标签和标题
#     ax.set_ylabel('Pearson\'s correlation coefficient', fontsize=20)
#
#     # 设置图例位置和样式
#     legend = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0, framealpha=0)  # 调整图例位置和去除边框
#
#     # 去除顶部和右侧边框线
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(axis='x', labelsize=20, colors='black', width=1)
#     ax.tick_params(axis='y', labelsize=20, colors='black', width=1)
#     # 自动调整x轴标签的间距
#     plt.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.9)  # 调整上边距
#
#     # 设置y轴范围
#     ax.set_ylim(0.6, 1.06)  # 调整y轴范围以显示小于1的数据并留出空白空间
#     # 设置图例文字大小
#     font_properties = FontProperties(size='large')
#     for text in legend.get_texts():
#         text.set_fontproperties(font_properties)
#     # 设置x轴刻度标签
#     ax.set_xticks(x)
#     ax.set_xticklabels(['2014 year', '2015 year', '2016 year', '2017 year', '2018 year'])
#     # 去除坐标轴刻度点
#     ax.tick_params(axis='both', which='both', bottom=False, left=False,top = False,right = False)
#     for spine in ax.spines.values():
#         spine.set_linewidth(2)  # Adjust the linewidth as desired
#     ax.tick_params(axis='both', which='both', width=2)
#     # plt.tight_layout()
#     plt.savefig('person.tif', dpi=300, format='tif', bbox_inches='tight', pad_inches=0)
#     # 显示图形
#     plt.show()
# 创建图形和轴
matplotlib.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(8.8, 4))
# matplotlib.rcParams.update({'font.size': 15})
# # 设置图形样式
# plt.style.use('seaborn-deep')  # 使用深色主题

# 绘制柱状图
rects1 = ax.bar(x - width/2, linear_regression_coefficients, width, label='Linear Regression', color='blue')
rects2 = ax.bar(x + width/2, gc_gru_coefficients, width, label='GC-GRU model', color='orange')

# 添加标签和标题
ax.set_ylabel('Pearson\'s correlation coefficient')


# 设置图例位置和样式
legend = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), borderaxespad=0, framealpha=0)  # 调整图例位置和去除边框

# 去除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.tick_params(axis='x', labelsize=20, colors='black', width=1)
# ax.tick_params(axis='y', labelsize=20, colors='black', width=1)
# 自动调整x轴标签的间距
plt.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.9)  # 调整上边距

# 设置y轴范围
ax.set_ylim(0.7, 1.09)  # 调整y轴范围以显示小于1的数据并留出空白空间
# 设置图例文字大小
# font_properties = FontProperties(size='large')
# for text in legend.get_texts():
#     text.set_fontproperties(font_properties)
# 设置x轴刻度标签
ax.set_xticks(x)
ax.set_xticklabels(['2014', '2015', '2016', '2017', '2018'])
ax.set_xlabel('Time (year)')
# plt.tight_layout()
plt.savefig('person2.tif',dpi=600, format='tif', bbox_inches='tight',pad_inches=0)
# 显示图形
plt.show()