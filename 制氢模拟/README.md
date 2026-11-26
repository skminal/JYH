# 制氢模拟

| 文件夹 | 内容 |
|---|---|
| `plasma_methanol_cfd/` | 甲醇中针–环电极 2.6 kV 放电：针尖气相来源（焦耳热 vs 放电）与起始判据。Gerris（VOF + 欧姆传导 + 焦耳热 + 相变 + 放电闭合）。<br>`results/`：动图 `needle_2p6kV.gif`、`needle_2p6kV_minimal.gif`，判据图，组会 PPT `甲醇针尖气相来源_组会汇报.pptx`；`NOTES.md`：工况、验证、结论与限制。 |
| `plasma_bubble_openfoam/` | 乙醇中氦等离子体气泡 2.0 kV 正/负极性复刻（初始气泡 → 颈缩 → 脱离），OpenFOAM v1912 自编求解器 plasmaInterFoam，含表面张力。<br>`results/plasma_bubble_2p0kV.gif`；`README.md`：工况、模型、结果对比与限制；`case_setup/`：三个算例的完整设置。 |
| `文献/` | Wang et al., *Matter* 7 (2024) 3024–3035 原文与补充视频。 |
