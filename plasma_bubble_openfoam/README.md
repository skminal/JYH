# 乙醇中氦等离子体气泡（2.0 kV，正/负极性）——OpenFOAM 复刻

参考：Wang et al., *Matter* 7 (2024) 3024–3035，Video S6（10000 fps）。

## 结果
- `results/plasma_bubble_2p0kV.gif`：0 kV / +2.0 kV / −2.0 kV，0–60 ms，含表面张力（σ = 22.14 mN/m）。
- `runs/*/metrics.json`：每 0.5 ms 的气泡顶高、底深、宽度、附着体积、脱离气泡体积与位置。

| | 颈缩断开 | 脱离体积 | 残留体积 |
|---|---|---|---|
| +2.0 kV（模拟） | ≈16 ms | 6.1 mm³ | 9.8 mm³ |
| −2.0 kV（模拟） | ≈12 ms | 8.4 mm³ | 7.3 mm³ |
| −2.0 kV（视频实测） | — | 7.7 mm³ | 9.2 mm³ |

## 工况
乙醇 ρ 791 kg/m³、μ 1.17 mPa·s、σ 22.14 mN/m；氦 ρ 0.1625 kg/m³、μ 1.99e-5 Pa·s；
石英毛细管 ID 0.3 / OD 0.5 mm，He 3 mL/min（管内平均流速 0.707 m/s）；轴对称楔形网格，r ≤ 5 mm，z ∈ [−3, 20] mm，
近区网格 50 µm；初始为视频测得的残留气泡（R = 1.54 mm，约 15 mm³）。

## 等离子体模型（唯象，参数对视频标定，不是预测）
求解器 `solver/plasmaInterFoam` = OpenFOAM v1912 的 interFoam + 三项力，只作用于与针尖相连的气体：
1. 把气体拉向针尖的中心 EHD 力：F0 = 40·ρ_l·g，作用半径 1.5 mm；
2. 顶部向外的界面应力（离子风）：60 Pa·cos²θ；
3. 随机界面应力（极性差别）：正极 5 Pa，负极 8 Pa。

## 限制
- 负极扰动 >25 Pa 时整个气泡被甩离毛细管，所以只能用 8 Pa，表面褶皱远弱于视频。
- 断开比视频快（视频正极约 52 ms）。
- 50 µm 网格，未做网格收敛。

## 复现
1. 用 Ubuntu 的 `openfoam` 包（v1912）与其源码包；`solver/ofenv.sh` 设置编译环境，在 `solver/plasmaInterFoam` 中运行 `wmake`。
2. `python3 case/make_case.py <dir> --mode pos --dx 50e-6 --kappa 40 --RE 1.5e-3 --wE 0.15e-3 --Rb0 1.54e-3 --Ppos 5 --Pup 60 --tend 0.06`
   （负极：`--mode neg --Pneg 8`；0 kV：`--mode off`），再 `case/run.sh <dir>`。
3. `post/metrics.py` 统计，`post/render_gif.py` 出图。
