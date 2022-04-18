import matplotlib.pyplot as plt

squares = [
    [0.77311668, 0.38363847, 0.63505706],
    [0.74788957, 0.52101156, 0.55336089],
    [0.66782278, 0.40200977, 0.59211279, 0.431345],
    [0.7097479, 0.35371519, 0.68981135, 0.2794211],
    [0.62393426, 0.55183142, 0.59728071],
    [0.78390659, 0.62697804, 0.70391519],
    [0.8992039, 0.51748103, 0.70103837],
    [0.69250282, 0.5921222, 0.46518866]
]
avedata = [
    0.5972707389779867,
    0.6074206737168589,
    0.5233225875191062,
    0.5081738848119612,
    0.5910154638788944,
    0.7049332720023974,
    0.7059077665538248,
    0.5832712277359483
]
values = ['水墨1', '水墨2', '彩墨1', '彩墨2', '油画1', '油画2', '水彩1', '水彩2']
plt.plot(values, [x[0] for x in squares], color='yellow', marker='*', linestyle='-', label='云水', linewidth=2)
plt.plot(values, [x[1] for x in squares], color='#00FFFF', marker='o', linestyle='-', label='山石/迎光面植被', linewidth=2)
plt.plot(values, [x[2] for x in squares], color='#0000FF', marker='s', linestyle='-', label='植被/背光面植被', linewidth=2)
plt.plot(values, [sum(x)/3 for x in squares], color='red', marker='v', linestyle='-', label='平均值', linewidth=2)
plt.plot(values[2:4], [squares[2][3], squares[3][3]], color='#00C800', marker='^', linestyle='-', label='细分割植被', linewidth=2)
plt.legend()
plt.ylim(0,1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.title("云水、山石、植被分类情况IOU对比图")
plt.xlabel("画作名称")
plt.ylabel("IOU")
plt.tick_params(axis='x', labelsize=10)
plt.savefig("iou.png")