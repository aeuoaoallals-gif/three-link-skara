import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import csv
import os
from ament_index_python.packages import get_package_share_directory

L1 = 0.37
L2 = 0.25
L3 = 0.20

@dataclass
class TrajectoryPoint:
    t: float
    q1: float
    q2: float
    q3: float

@dataclass
class Segment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    duration: float

def ik(x: float, y: float, phi: float, q_prev: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x_w = x - L3 * np.cos(phi)
    y_w = y - L3 * np.sin(phi)

    r_w = np.hypot(x_w, y_w)
    if r_w > (L1 + L2 - 0.001) or r_w < (abs(L1 - L2) + 0.001):
        return q_prev

    D = (x_w**2 + y_w**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)

    q2_candidates = [
        np.arctan2(np.sqrt(1 - D**2), D),
        np.arctan2(-np.sqrt(1 - D**2), D)
    ]
    q2_new = min(q2_candidates, key=lambda q: abs(q - q_prev[1]))
    q1_new = np.arctan2(y_w, x_w) - np.arctan2(L2 * np.sin(q2_new), L1 + L2 * np.cos(q2_new))
    q3_new = phi - q1_new - q2_new

    return q1_new, q2_new, q3_new

def compute_phi(x: float, y: float) -> float:
    """Вычисляет phi так, чтобы запястье было на оптимальном расстоянии от базы"""
    r_target = np.hypot(x, y)
    if r_target > 1e-6:
        # Направление от базы к цели + pi (запястье дальше от базы)
        return np.arctan2(y, x) + np.pi
    else:
        return 0.0

def generate_trajectory(segments: List[Segment], dt: float = 0.001) -> List[TrajectoryPoint]:
    points: List[TrajectoryPoint] = []
    current_t = 0.0

    # Начальные углы для первой точки (0, 0.30)
    start_x, start_y = segments[0].start
    phi_start = compute_phi(start_x, start_y)
    
    # Вычисляем начальные углы через IK
    x_w = start_x - L3 * np.cos(phi_start)
    y_w = start_y - L3 * np.sin(phi_start)
    D = (x_w**2 + y_w**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)
    q2_start = np.arctan2(np.sqrt(1 - D**2), D)
    q1_start = np.arctan2(y_w, x_w) - np.arctan2(L2 * np.sin(q2_start), L1 + L2 * np.cos(q2_start))
    q3_start = phi_start - q1_start - q2_start

    print(f"Начальные углы: q1={q1_start:.4f}, q2={q2_start:.4f}, q3={q3_start:.4f}")
    print(f"phi_start={phi_start:.4f}")

    q1, q2, q3 = q1_start, q2_start, q3_start
    points.append(TrajectoryPoint(t=0.0, q1=q1, q2=q2, q3=q3))

    for seg_idx, seg in enumerate(segments):
        x_start, y_start = seg.start
        x_end, y_end = seg.end
        T = seg.duration

        L = np.hypot(x_end - x_start, y_end - y_start)
        v = L / T if T > 0 else 0.0

        ux = (x_end - x_start) / L if L > 1e-12 else 0.0
        uy = (y_end - y_start) / L if L > 1e-12 else 0.0

        xdot = v * ux
        ydot = v * uy

        n_steps = max(1, int(round(T / dt)))
        t_seg = np.linspace(0, T, n_steps + 1)

        for i, tau in enumerate(t_seg):
            if i == 0 and seg_idx == 0:
                continue

            t_abs = current_t + tau
            x = x_start + xdot * tau
            y = y_start + ydot * tau

            # Адаптивный phi для каждой точки
            phi = compute_phi(x, y)

            q1, q2, q3 = ik(x, y, phi, (q1, q2, q3))
            points.append(TrajectoryPoint(t=t_abs, q1=q1, q2=q2, q3=q3))

        current_t += T

    q1_arr = np.unwrap([p.q1 for p in points])
    q2_arr = np.unwrap([p.q2 for p in points])
    q3_arr = np.unwrap([p.q3 for p in points])

    for i in range(len(points)):
        points[i] = TrajectoryPoint(t=points[i].t, q1=float(q1_arr[i]), q2=float(q2_arr[i]), q3=float(q3_arr[i]))

    return points

STAR_POINTS: List[Tuple[float, float]] = [
    ( 0.0000,  0.3000),
    ( 0.0980,  0.1348),
    ( 0.2853,  0.0929),
    ( 0.1589, -0.0458),
    ( 0.1767, -0.2425),
    ( 0.0000, -0.1700),
    (-0.1767, -0.2425),
    (-0.1589, -0.0458),
    (-0.2853,  0.0929),
    (-0.0980,  0.1348),
]

def main():
    seg_duration = 3.0
    segments = [Segment(STAR_POINTS[i], STAR_POINTS[(i + 1) % len(STAR_POINTS)], seg_duration) for i in range(len(STAR_POINTS))]

    print(f"Сформировано сегментов: {len(segments)}")
    print(f"L1={L1} м, L2={L2} м, L3={L3} м")

    dt = 0.001
    points = generate_trajectory(segments, dt)

    time = [p.t for p in points]
    q1 = [p.q1 for p in points]
    q2 = [p.q2 for p in points]
    q3 = [p.q3 for p in points]

    print(f"\nНачальные углы: q1={q1[0]:.4f}, q2={q2[0]:.4f}, q3={q3[0]:.4f}")
    print(f"Мин/Макс |q2|: {np.min(np.abs(q2)):.4f} / {np.max(np.abs(q2)):.4f} рад")

    pkg_scara_sim = get_package_share_directory('scara_sim')
    data_dir = os.path.join(pkg_scara_sim, 'data')
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'trajectory_angles.csv')

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'q1', 'q2', 'q3'])
        for p in points:
            writer.writerow([p.t, p.q1, p.q2, p.q3])

    print(f"\nCSV сохранён в {file_path}")
    print(f"Всего точек: {len(points)}")
    print(f"Длительность: {points[-1].t:.2f} с")

    fig = plt.figure(figsize=(14, 10))
    ax_xy = fig.add_subplot(2, 2, 1)
    x_traj = [L1*np.cos(q1[i]) + L2*np.cos(q1[i]+q2[i]) + L3*np.cos(q1[i]+q2[i]+q3[i]) for i in range(len(points))]
    y_traj = [L1*np.sin(q1[i]) + L2*np.sin(q1[i]+q2[i]) + L3*np.sin(q1[i]+q2[i]+q3[i]) for i in range(len(points))]
    ax_xy.plot(x_traj, y_traj, 'b-', linewidth=1.5, label='траектория')

    sx = [p[0] for p in STAR_POINTS] + [STAR_POINTS[0][0]]
    sy = [p[1] for p in STAR_POINTS] + [STAR_POINTS[0][1]]
    ax_xy.plot(sx, sy, 'ro', markersize=6, label='вершины')
    ax_xy.plot(sx, sy, 'r--', alpha=0.4)
    ax_xy.set_aspect('equal')
    ax_xy.set_title(f'Контур звезды (L1={L1}, L2={L2}, L3={L3})')
    ax_xy.grid(True)
    ax_xy.legend()

    fig.add_subplot(2, 2, 2).plot(time, q1)
    plt.xlabel('t'); plt.ylabel('q1'); plt.grid(True)

    fig.add_subplot(2, 2, 3).plot(time, q2)
    plt.xlabel('t'); plt.ylabel('q2'); plt.grid(True)

    fig.add_subplot(2, 2, 4).plot(time, q3)
    plt.xlabel('t'); plt.ylabel('q3'); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()