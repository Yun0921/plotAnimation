import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import os

# 讀取 CSV 檔案
csv_path = "demo_patient_07_predictions_last20.csv"
df = pd.read_csv(csv_path)

# 建立畫布與 gridspec
fig = plt.figure(figsize=(16, 12))  # 加高圖片以容納更大的間距
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])  # 確保三個子圖等高

# 添加角度圖
ax_roll = fig.add_subplot(gs[0])    # 第一排
ax_pitch = fig.add_subplot(gs[1])   # 第二排
ax_yaw = fig.add_subplot(gs[2])     # 第三排

# 創建角度圖的線條
roll_true, = ax_roll.plot([], [], 'r-', label='True Roll')
roll_pred, = ax_roll.plot([], [], 'b--', label='Predicted Roll')
pitch_true, = ax_pitch.plot([], [], 'r-', label='True Pitch')
pitch_pred, = ax_pitch.plot([], [], 'b--', label='Predicted Pitch')
yaw_true, = ax_yaw.plot([], [], 'r-', label='True Yaw')
yaw_pred, = ax_yaw.plot([], [], 'b--', label='Predicted Yaw')

# 設置角度圖的標籤和範圍
for ax, title in zip([ax_roll, ax_pitch, ax_yaw], ['Roll', 'Pitch', 'Yaw']):
    ax.set_xlabel('Pair Index')
    ax.set_ylabel('Angle (degree)')
    ax.set_title(f'{title} Angle')
    ax.grid(True)
    ax.legend()
    # 設置 x 軸為整數刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 準備數據
data_length = len(df) - 1  # 因為要配對，所以長度減1

# 設置角度圖的 x 軸範圍
for ax in [ax_roll, ax_pitch, ax_yaw]:
    ax.set_xlim(0, data_length-1)

# 計算角度範圍
angle_data = {
    'roll': {'true': [], 'pred': []},
    'pitch': {'true': [], 'pred': []},
    'yaw': {'true': [], 'pred': []}
}

for i in range(data_length):
    # 真實角度
    angle_data['roll']['true'].append(df.loc[i + 1, 'gt_roll'])
    angle_data['pitch']['true'].append(df.loc[i + 1, 'gt_pitch'])
    angle_data['yaw']['true'].append(df.loc[i + 1, 'gt_yaw'])
    # 預測角度
    angle_data['roll']['pred'].append(df.loc[i, 'pred_roll'])
    angle_data['pitch']['pred'].append(df.loc[i, 'pred_pitch'])
    angle_data['yaw']['pred'].append(df.loc[i, 'pred_yaw'])

# 設置角度圖的 y 軸範圍
margin = 10  # 增加 10 度的邊界
for ax, angle_type in zip([ax_roll, ax_pitch, ax_yaw], ['roll', 'pitch', 'yaw']):
    true_min = min(angle_data[angle_type]['true'])
    true_max = max(angle_data[angle_type]['true'])
    pred_min = min(angle_data[angle_type]['pred'])
    pred_max = max(angle_data[angle_type]['pred'])
    y_min = min(true_min, pred_min) - margin
    y_max = max(true_max, pred_max) + margin
    ax.set_ylim(y_min, y_max)

# 調整子圖間距
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.5)  # 增加 hspace 來加大間距

def update(frame):
    # 使用整數作為 x 軸值
    x = list(range(frame + 1))
    
    # Roll
    roll_true.set_data(x, angle_data['roll']['true'][:frame + 1])
    roll_pred.set_data(x, angle_data['roll']['pred'][:frame + 1])
    
    # Pitch
    pitch_true.set_data(x, angle_data['pitch']['true'][:frame + 1])
    pitch_pred.set_data(x, angle_data['pitch']['pred'][:frame + 1])
    
    # Yaw
    yaw_true.set_data(x, angle_data['yaw']['true'][:frame + 1])
    yaw_pred.set_data(x, angle_data['yaw']['pred'][:frame + 1])
    
    return [roll_true, roll_pred, pitch_true, pitch_pred, yaw_true, yaw_pred]

# 創建動畫
ani = FuncAnimation(fig, update, frames=data_length, interval=1000, blit=True)

# 確保輸出目錄存在
output_dir = "animation_output"
os.makedirs(output_dir, exist_ok=True)

# 保存為 GIF
gif_path = os.path.join(output_dir, "angle_animation.gif")
print(f"正在生成 GIF... 請稍候...")
ani.save(gif_path, writer='pillow', fps=1)
print(f"GIF 已保存至: {gif_path}")

plt.show()
