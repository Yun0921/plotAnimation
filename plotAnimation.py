import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os

# 讀取 CSV 檔案
csv_path = "demo_patient_07_predictions_last20.csv"
df = pd.read_csv(csv_path)

# 準備資料
data = []
for i in range(len(df) - 1):
    at1 = df.loc[i, ["gt_x", "gt_y", "gt_z"]].values
    at2 = df.loc[i + 1, ["gt_x", "gt_y", "gt_z"]].values
    at2_pred = df.loc[i, ["pred_x", "pred_y", "pred_z"]].values
    data.append({"at1": at1, "at2": at2, "at2_pred": at2_pred})

# 計算完整軌跡
full_trajectory = np.array([d["at1"] for d in data])

# 計算範圍
all_points = np.array([d["at1"].tolist() + d["at2"].tolist() + d["at2_pred"].tolist() for d in data]).reshape(-1, 3)
mins = all_points.min(axis=0)
maxs = all_points.max(axis=0)

# 計算每個維度的範圍大小
ranges = maxs - mins
# 增加邊界空間（每邊增加 20% 的範圍）
padding = ranges * 0.2
mins = mins - padding
maxs = maxs + padding

# 建立畫布與 gridspec
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1])  # 3 row, 2 col layout

# 子圖分配
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')   # 左半整排
ax_xy = fig.add_subplot(gs[0, 1])                    # 右上
ax_yz = fig.add_subplot(gs[1, 1])                    # 右中
ax_xz = fig.add_subplot(gs[2, 1])                    # 右下

# 建立角度圖的新視窗
fig_angles = plt.figure(figsize=(16, 8))
gs_angles = gridspec.GridSpec(3, 1)

# 添加角度圖
ax_roll = fig_angles.add_subplot(gs_angles[0])    # 第一排
ax_pitch = fig_angles.add_subplot(gs_angles[1])   # 第二排
ax_yaw = fig_angles.add_subplot(gs_angles[2])     # 第三排

# 設定 3D 圖
ax_3d.set_xlim(mins[0], maxs[0])
ax_3d.set_ylim(mins[1], maxs[1])
ax_3d.set_zlim(mins[2], maxs[2])
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.set_title("3D View")

# 調整主視窗子圖間距
plt.figure(fig.number)  # 切換到主視窗
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

# 調整角度圖視窗的子圖間距
plt.figure(fig_angles.number)  # 切換到角度圖視窗
plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.3)

# 設定平面圖
for ax, title, x_label, y_label, x_idx, y_idx in zip(
    [ax_xy, ax_yz, ax_xz],
    ["XY Plane", "YZ Plane", "XZ Plane"],
    ["X", "Y", "X"],
    ["Y", "Z", "Z"],
    [0, 1, 0],
    [1, 2, 2]
):
    ax.set_xlim(mins[x_idx], maxs[x_idx])
    ax.set_ylim(mins[y_idx], maxs[y_idx])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

# 建立所有動畫物件
def create_plot_objects(ax3d, ax2d, x_idx, y_idx):
    # 3D 物件
    p1_3d, = ax3d.plot([], [], [], 'ko', markersize=4, label='AT1')
    p2_3d, = ax3d.plot([], [], [], 'ro', markersize=4, label='True AT2')
    pp_3d, = ax3d.plot([], [], [], 'bo', markersize=4, label='Pred AT2')
    l12_3d = ax3d.plot([], [], [], 'r-', label='True Path')[0]
    lp_3d = ax3d.plot([], [], [], 'b:', label='Pred Path')[0]
    # 添加已走過和未走過的軌跡
    past_trail_3d, = ax3d.plot([], [], [], 'k-', alpha=0.5, label='Past Trail')
    future_trail_3d, = ax3d.plot([], [], [], 'k:', alpha=0.5, label='Future Trail')

    # 2D 物件 (不需要標籤)
    p1_2d, = ax2d.plot([], [], 'ko', markersize=4)
    p2_2d, = ax2d.plot([], [], 'ro', markersize=4)
    pp_2d, = ax2d.plot([], [], 'bo', markersize=4)
    l12_2d, = ax2d.plot([], [], 'r-')
    lp_2d, = ax2d.plot([], [], 'b:')
    past_trail_2d, = ax2d.plot([], [], 'k-', alpha=0.5)
    future_trail_2d, = ax2d.plot([], [], 'k:', alpha=0.5)

    return (p1_3d, p2_3d, pp_3d, l12_3d, lp_3d, past_trail_3d, future_trail_3d,
            p1_2d, p2_2d, pp_2d, l12_2d, lp_2d, past_trail_2d, future_trail_2d)

objs_xy = create_plot_objects(ax_3d, ax_xy, 0, 1)
objs_yz = create_plot_objects(ax_3d, ax_yz, 1, 2)
objs_xz = create_plot_objects(ax_3d, ax_xz, 0, 2)

# 只在第一組物件中添加圖例
ax_3d.legend(
    handles=[
        objs_xy[0], objs_xy[1], objs_xy[2],  # 點
        objs_xy[3], objs_xy[4],              # 線
        objs_xy[5], objs_xy[6]               # 軌跡
    ],
    loc='upper right',
    framealpha=0.8,
    edgecolor='white',
    facecolor='white',
    fontsize=9
)

# 創建角度圖的線條
roll_true, = ax_roll.plot([], [], 'r-', label='True Roll')
roll_pred, = ax_roll.plot([], [], 'b--', label='Predicted Roll')
pitch_true, = ax_pitch.plot([], [], 'r-', label='True Pitch')
pitch_pred, = ax_pitch.plot([], [], 'b--', label='Predicted Pitch')
yaw_true, = ax_yaw.plot([], [], 'r-', label='True Yaw')
yaw_pred, = ax_yaw.plot([], [], 'b--', label='Predicted Yaw')

# 設置角度圖的標籤和範圍
for ax, title in zip([ax_roll, ax_pitch, ax_yaw], ['Roll', 'Pitch', 'Yaw']):
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degree)')
    ax.set_title(f'{title} Angle')
    ax.grid(True)
    ax.legend()

# 設置角度圖的 x 軸範圍
for ax in [ax_roll, ax_pitch, ax_yaw]:
    ax.set_xlim(0, len(data))

# 計算角度範圍
angle_data = {
    'roll': {'true': [], 'pred': []},
    'pitch': {'true': [], 'pred': []},
    'yaw': {'true': [], 'pred': []}
}

for i in range(len(df) - 1):
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

# 更新函數
def update(frame):
    at1 = data[frame]["at1"]
    at2 = data[frame]["at2"]
    at2_pred = data[frame]["at2_pred"]

    updates = []

    for (p1_3d, p2_3d, pp_3d, l12_3d, lp_3d, past_trail_3d, future_trail_3d,
         p1_2d, p2_2d, pp_2d, l12_2d, lp_2d, past_trail_2d, future_trail_2d), (x_idx, y_idx) in zip(
        [objs_xy, objs_yz, objs_xz], [(0, 1), (1, 2), (0, 2)]
    ):
        # 3D 更新一次就夠
        if x_idx == 0 and y_idx == 1:
            p1_3d.set_data([at1[0]], [at1[1]])
            p1_3d.set_3d_properties([at1[2]])
            p2_3d.set_data([at2[0]], [at2[1]])
            p2_3d.set_3d_properties([at2[2]])
            pp_3d.set_data([at2_pred[0]], [at2_pred[1]])
            pp_3d.set_3d_properties([at2_pred[2]])
            l12_3d.set_data([at1[0], at2[0]], [at1[1], at2[1]])
            l12_3d.set_3d_properties([at1[2], at2[2]])
            lp_3d.set_data([at1[0], at2_pred[0]], [at1[1], at2_pred[1]])
            lp_3d.set_3d_properties([at1[2], at2_pred[2]])
            
            # 更新軌跡
            if frame > 0:
                # 已走過的路徑（實線）
                past_trail_3d.set_data(full_trajectory[:frame+1, 0], full_trajectory[:frame+1, 1])
                past_trail_3d.set_3d_properties(full_trajectory[:frame+1, 2])
            # 未走過的路徑（虛線）
            future_trail_3d.set_data(full_trajectory[frame:, 0], full_trajectory[frame:, 1])
            future_trail_3d.set_3d_properties(full_trajectory[frame:, 2])
            
            updates += [p1_3d, p2_3d, pp_3d, l12_3d, lp_3d, past_trail_3d, future_trail_3d]

        # 2D 更新
        p1_2d.set_data([at1[x_idx]], [at1[y_idx]])
        p2_2d.set_data([at2[x_idx]], [at2[y_idx]])
        pp_2d.set_data([at2_pred[x_idx]], [at2_pred[y_idx]])
        l12_2d.set_data([at1[x_idx], at2[x_idx]], [at1[y_idx], at2[y_idx]])
        lp_2d.set_data([at1[x_idx], at2_pred[x_idx]], [at1[y_idx], at2_pred[y_idx]])
        
        # 更新 2D 軌跡
        if frame > 0:
            past_trail_2d.set_data(full_trajectory[:frame+1, x_idx], full_trajectory[:frame+1, y_idx])
        future_trail_2d.set_data(full_trajectory[frame:, x_idx], full_trajectory[frame:, y_idx])
        
        updates += [p1_2d, p2_2d, pp_2d, l12_2d, lp_2d, past_trail_2d, future_trail_2d]

    # 更新角度圖
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
    
    updates += [roll_true, roll_pred, pitch_true, pitch_pred, yaw_true, yaw_pred]

    return updates

# 動畫與顯示
ani = FuncAnimation(fig, update, frames=len(data), interval=1000, blit=True)
ani_angles = FuncAnimation(fig_angles, update, frames=len(data), interval=1000, blit=True)

# 確保輸出目錄存在
output_dir = "animation_output"
os.makedirs(output_dir, exist_ok=True)

# 保存為 GIF（只保存主視窗的動畫）
gif_path = os.path.join(output_dir, "prediction_animation.gif")
print(f"正在生成 GIF... 請稍候...")
ani.save(gif_path, writer='pillow', fps=1)
print(f"GIF 已保存至: {gif_path}")

plt.show()
