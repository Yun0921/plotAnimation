import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import os

def create_trajectory_animation(df, output_path):
    # 準備資料
    data = []
    for i in range(len(df) - 1):
        at1 = df.loc[i, ["gt_x", "gt_y", "gt_z"]].values
        at2 = df.loc[i + 1, ["gt_x", "gt_y", "gt_z"]].values
        at2_pred = df.loc[i, ["pred_x", "pred_y", "pred_z"]].values
        data.append({"at1": at1, "at2": at2, "at2_pred": at2_pred})

    # 計算完整軌跡
    full_trajectory = np.array([d["at1"] for d in data] + [data[-1]["at2"]])


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

    # 設定 3D 圖
    ax_3d.set_xlim(mins[0], maxs[0])
    ax_3d.set_ylim(mins[1], maxs[1])
    ax_3d.set_zlim(mins[2], maxs[2])
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title("3D View")

    # 設定平面圖
    for ax, title, x_label, y_label, x_idx, y_idx in zip(
        [ax_xy, ax_yz, ax_xz],
        ["XY Plane", "YZ Plane", "XZ Plane"],
        ["X (mm)", "Y (mm)", "X (mm)"],
        ["Y (mm)", "Z (mm)", "Z (mm)"],
        [0, 1, 0],
        [1, 2, 2]
    ):
        ax.set_xlim(mins[x_idx], maxs[x_idx])
        ax.set_ylim(mins[y_idx], maxs[y_idx])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    def create_plot_objects(ax3d, ax2d, x_idx, y_idx):
        # 3D 物件
        p1_3d, = ax3d.plot([], [], [], 'ko', markersize=4, label='AT1')
        p2_3d, = ax3d.plot([], [], [], 'ro', markersize=4, label='True AT2')
        pp_3d, = ax3d.plot([], [], [], 'bo', markersize=4, label='Pred AT2')
        l12_3d = ax3d.plot([], [], [], 'r-', label='True Path')[0]
        lp_3d = ax3d.plot([], [], [], 'b:', label='Pred Path')[0]
        # 添加 at1 軌跡
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

    # 調整子圖間距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)

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

        return updates

    # 創建動畫
    ani = FuncAnimation(fig, update, frames=len(data), interval=1000, blit=True)

    # 保存為 GIF
    print(f"正在生成軌跡動畫 GIF... 請稍候...")
    ani.save(output_path, writer='pillow', fps=1)
    print(f"軌跡動畫 GIF 已保存至: {output_path}")
    plt.close()

def create_angle_animation(df, output_path):
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
        
        # 設置主網格和次網格
        ax.grid(True, which='major', linewidth=0.8, linestyle='-', color='#CCCCCC')
        ax.grid(True, which='minor', linewidth=0.5, linestyle=':', color='#DDDDDD')
        ax.minorticks_on()  # 啟用次要刻度
        
        # 強調 y=0 的橫軸
        ax.axhline(y=0, color='k', linewidth=1.5)
        
        # 設置邊框顏色和粗細
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        ax.legend()
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

    # 保存為 GIF
    print(f"正在生成角度動畫 GIF... 請稍候...")
    ani.save(output_path, writer='pillow', fps=1)
    print(f"角度動畫 GIF 已保存至: {output_path}")
    plt.close()

def create_combined_animation(df, output_path):
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
    padding = ranges * 0.2
    mins = mins - padding
    maxs = maxs + padding

    # 建立畫布與 gridspec
    fig = plt.figure(figsize=(20, 16))
    
    # 創建主網格：2行，上面用於軌跡圖，下面用於角度圖 (3:1 比例)
    main_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # 上半部分的網格，用於軌跡圖 (1列2行，左邊3D圖，右邊三個平面圖)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=main_gs[0], width_ratios=[1.2, 1])
    
    # 右側平面圖的子網格 (3列1行)
    right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=top_gs[1], hspace=0.3)
    
    # 下半部分的網格，用於角度圖
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=main_gs[1], wspace=0.3)

    # 分配軌跡圖的子圖
    ax_3d = fig.add_subplot(top_gs[0], projection='3d')   # 左側3D圖
    ax_xy = fig.add_subplot(right_gs[0])                  # 右側上方
    ax_yz = fig.add_subplot(right_gs[1])                  # 右側中間
    ax_xz = fig.add_subplot(right_gs[2])                  # 右側下方

    # 分配角度圖的子圖
    ax_roll = fig.add_subplot(bottom_gs[0])    # 下排左
    ax_pitch = fig.add_subplot(bottom_gs[1])   # 下排中
    ax_yaw = fig.add_subplot(bottom_gs[2])     # 下排右

    # 設定 3D 圖
    ax_3d.set_xlim(mins[0], maxs[0])
    ax_3d.set_ylim(mins[1], maxs[1])
    ax_3d.set_zlim(mins[2], maxs[2])
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_title("3D View")

    # 設定平面圖
    for ax, title, x_label, y_label, x_idx, y_idx in zip(
        [ax_xy, ax_yz, ax_xz],
        ["XY Plane", "YZ Plane", "XZ Plane"],
        ["X (mm)", "Y (mm)", "X (mm)"],
        ["Y (mm)", "Z (mm)", "Z (mm)"],
        [0, 1, 0],
        [1, 2, 2]
    ):
        ax.set_xlim(mins[x_idx], maxs[x_idx])
        ax.set_ylim(mins[y_idx], maxs[y_idx])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    # 創建軌跡圖的物件
    def create_plot_objects(ax3d, ax2d, x_idx, y_idx):
        # 3D 物件
        p1_3d, = ax3d.plot([], [], [], 'ko', markersize=4, label='AT1')
        p2_3d, = ax3d.plot([], [], [], 'ro', markersize=4, label='True AT2')
        pp_3d, = ax3d.plot([], [], [], 'bo', markersize=4, label='Pred AT2')
        l12_3d = ax3d.plot([], [], [], 'r-', label='True Path')[0]
        lp_3d = ax3d.plot([], [], [], 'b:', label='Pred Path')[0]
        # 添加 at1 軌跡
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
        
        # 設置主網格和次網格
        ax.grid(True, which='major', linewidth=0.8, linestyle='-', color='#CCCCCC')
        ax.grid(True, which='minor', linewidth=0.5, linestyle=':', color='#DDDDDD')
        ax.minorticks_on()  # 啟用次要刻度
        
        # 強調 y=0 的橫軸
        ax.axhline(y=0, color='k', linewidth=1.5)
        
        # 設置邊框顏色和粗細
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 準備角度數據
    data_length = len(df) - 1
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

    # 設置角度圖的 x 軸範圍
    for ax in [ax_roll, ax_pitch, ax_yaw]:
        ax.set_xlim(0, data_length-1)

    # 設置角度圖的 y 軸範圍
    margin = 10
    for ax, angle_type in zip([ax_roll, ax_pitch, ax_yaw], ['roll', 'pitch', 'yaw']):
        true_min = min(angle_data[angle_type]['true'])
        true_max = max(angle_data[angle_type]['true'])
        pred_min = min(angle_data[angle_type]['pred'])
        pred_max = max(angle_data[angle_type]['pred'])
        y_min = min(true_min, pred_min) - margin
        y_max = max(true_max, pred_max) + margin
        ax.set_ylim(y_min, y_max)

    # 只在 3D 圖中添加圖例
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

    # 調整子圖間距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    def update(frame):
        updates = []
        
        # 更新軌跡圖
        at1 = data[frame]["at1"]
        at2 = data[frame]["at2"]
        at2_pred = data[frame]["at2_pred"]

        for (p1_3d, p2_3d, pp_3d, l12_3d, lp_3d, past_trail_3d, future_trail_3d,
             p1_2d, p2_2d, pp_2d, l12_2d, lp_2d, past_trail_2d, future_trail_2d), (x_idx, y_idx) in zip(
            [objs_xy, objs_yz, objs_xz], [(0, 1), (1, 2), (0, 2)]
        ):
            # 3D 更新
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
                
                if frame > 0:
                    past_trail_3d.set_data(full_trajectory[:frame+1, 0], full_trajectory[:frame+1, 1])
                    past_trail_3d.set_3d_properties(full_trajectory[:frame+1, 2])
                future_trail_3d.set_data(full_trajectory[frame:, 0], full_trajectory[frame:, 1])
                future_trail_3d.set_3d_properties(full_trajectory[frame:, 2])
                
                updates += [p1_3d, p2_3d, pp_3d, l12_3d, lp_3d, past_trail_3d, future_trail_3d]

            # 2D 更新
            p1_2d.set_data([at1[x_idx]], [at1[y_idx]])
            p2_2d.set_data([at2[x_idx]], [at2[y_idx]])
            pp_2d.set_data([at2_pred[x_idx]], [at2_pred[y_idx]])
            l12_2d.set_data([at1[x_idx], at2[x_idx]], [at1[y_idx], at2[y_idx]])
            lp_2d.set_data([at1[x_idx], at2_pred[x_idx]], [at1[y_idx], at2_pred[y_idx]])
            
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

    # 創建動畫
    ani = FuncAnimation(fig, update, frames=len(data), interval=1000, blit=True)

    # 保存為 GIF
    print(f"正在生成組合動畫 GIF... 請稍候...")
    ani.save(output_path, writer='pillow', fps=1)
    print(f"組合動畫 GIF 已保存至: {output_path}")
    plt.close()

def main():
    # 設定輸入和輸出目錄
    input_dir = "demo_outputs"
    output_dir = "animation_results"
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍歷所有子目錄
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)
        
        # 確保是目錄
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"\n處理目錄: {subdir}")
        
        # 🔻這裡不再建立子資料夾
        # subdir_output = os.path.join(output_dir, subdir)
        # os.makedirs(subdir_output, exist_ok=True)
        
        # 獲取該子目錄下的所有 CSV 文件
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        
        # 處理每個 CSV 文件
        for csv_file in csv_files:
            print(f"處理文件: {csv_file}")
            
            csv_path = os.path.join(subdir_path, csv_file)
            df = pd.read_csv(csv_path)
            
            output_filename = f"{subdir}_{os.path.splitext(csv_file)[0]}_animation.gif"
            combined_path = os.path.join(output_dir, output_filename)
            
            create_combined_animation(df, combined_path)
            
            print(f"已生成動畫: {output_filename}")


if __name__ == "__main__":
    main() 