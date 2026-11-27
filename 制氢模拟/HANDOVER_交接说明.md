# 交接说明：制氢模拟（等离子体气泡 / 甲醇针电极放电）

本文件汇总本项目在云端会话里的全部工作记忆，供本地会话（16 核电脑）直接接手。
写于 2026-11-27。仓库：`skminal/JYH`，分支 `claude/gifted-ride-5wfge9`，所有内容在 `制氢模拟/` 下。

---

## 0. 用户偏好（务必遵守）

- 中文交流；回答只要干货，不要无用描述。
- 算完**直接发结果文件**（GIF/PPT/PDF），不要发压缩包（用户打不开 zip），不要反复汇报进度、不要问“还要多久”类问题的答复冗长。
- 额度重置/会话恢复后，**不要重复检查已完成部分**，直接继续没做完的。
- 所有文件放在 `制氢模拟/` 文件夹。
- 模拟**必须包含表面张力**（用户自己的旧模拟没有表面张力，这是重点）。
- GIF 版式：按用户模板——多栏并排，每栏上方粗体电压标签（如 `0 kV`、`+2.0 kV`、`−2.0 kV`），坐标 x ∈ [−5, 5] mm、z ∈ [−3, 20] mm，管口处橙色虚线，每栏左上小字 `t = x.xxx ms`，最右上红色粗体总时间 `t = x.x ms`；气相蓝色。质量要高于参考图。
- 另一种用户要求过的极简版式（甲醇项目）：去掉标题/说明文字，坐标轴文字用 Times New Roman（本机无则 Liberation Serif），规范上下标，Origin 风格（刻度朝内、四边框）。
- PPT：不用模板，只要干货，页数按要求（6 页）。
- 实验者：张伟（论文 Wang, Zhang et al., *Matter* 7 (2024) 3024–3035）。

---

## 1. 两个子项目

| 文件夹 | 内容 | 状态 |
|---|---|---|
| `plasma_methanol_cfd/` | 甲醇针–环电极 2.6 kV：针尖气相是焦耳热还是放电产生？起始判据。Gerris（VOF+欧姆传导+焦耳热+相变+放电闭合）。 | 已完成：两版 GIF、判据图、组会 PPT、`NOTES.md` |
| `plasma_bubble_openfoam/` | 乙醇中氦等离子体气泡 2.0 kV 正/负极性复刻（残留→颈缩→脱离），OpenFOAM v1912 自编求解器 `plasmaInterFoam`。 | v1（论文几何）已完成；**v2（用户新几何）调参进行中**（见第 5 节） |
| `文献/` | 论文 PDF + 补充视频 mmc3–mmc7 | — |

---

## 2. OpenFOAM 子项目：环境搭建（本地照做）

云端是 Ubuntu 24.04、4 核。本地 16 核若是 Windows，建议 **WSL2 + Ubuntu 24.04**，在 WSL 里执行：

```bash
sudo apt-get update
sudo apt-get install -y openfoam openfoam-examples gcc g++ make flex bison zlib1g-dev python3-pip
pip install numpy scipy matplotlib pillow
# OpenFOAM v1912 的 Ubuntu 包没有头文件和 wmake，需要源码包编译自定义求解器：
mkdir -p ~/ofsrc && cd ~/ofsrc
curl -O http://archive.ubuntu.com/ubuntu/pool/universe/o/openfoam/openfoam_1912.200626.orig.tar.bz2
curl -O http://archive.ubuntu.com/ubuntu/pool/universe/o/openfoam/openfoam_1912.200626-2build3.debian.tar.xz
tar xjf openfoam_1912.200626.orig.tar.bz2 && tar xJf openfoam_1912.200626-2build3.debian.tar.xz
cd openfoam-OpenFOAM-v1912.200626 && for p in $(cat ../debian/patches/series); do patch -p1 < ../debian/patches/$p; done
```

编译环境脚本（`solver/ofenv.sh` 是云端路径版本，本地改成下面这样）：

```bash
export FOAM_INST_DIR=$HOME/ofsrc
. $HOME/ofsrc/openfoam-OpenFOAM-v1912.200626/etc/bashrc WM_COMPILER_TYPE=system WM_MPLIB=SYSTEMOPENMPI
export FOAM_LIBBIN=/usr/lib                 # 链接到 apt 装的二进制库
export FOAM_USER_APPBIN=$HOME/ofuser/bin
export FOAM_USER_LIBBIN=$HOME/ofuser/lib
export PATH=$FOAM_USER_APPBIN:$WM_PROJECT_DIR/wmake:/usr/bin:$PATH
export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:/usr/lib/openmpi/lib:/usr/lib:$LD_LIBRARY_PATH
```

然后：

```bash
source ofenv.sh
(cd $WM_PROJECT_DIR/wmake/src && make)          # 编译 wmake 工具
(cd $FOAM_SRC && wmakeLnIncludeAll .)            # 生成头文件链接
cp -r 制氢模拟/plasma_bubble_openfoam/solver/plasmaInterFoam ~/ofuser/src/
cp -r $FOAM_SOLVERS/multiphase/VoF ~/ofuser/src/VoF   # interFoam 依赖 ../VoF
cd ~/ofuser/src/plasmaInterFoam && wmake          # 得到 $FOAM_USER_APPBIN/plasmaInterFoam
```

注意：仓库里 `solver/plasmaInterFoam/` 已含 `Make/files`（EXE = `$(FOAM_USER_APPBIN)/plasmaInterFoam`）、`Make/options`（`-I../VoF`）。

### 已知坑
- Ubuntu 包的 functionObject（sampling、`postProcess -func`）报 `IOstream "sha1"` 错误：**不要在 controlDict 里用 functions/#include sampling**，后处理用 Python 直接读 ASCII 场。
- 示例文件在 `/usr/share/doc` 被精简镜像过滤，需 `apt-get download openfoam-examples && dpkg-deb -x`。
- scotch 分解在该包里是空壳：并行用 `method simple;`。
- `run.sh` 里路径写的是云端 scratchpad，本地需改 `S=` 那一行或直接 `source ofenv.sh`。

### 16 核并行
```bash
# 在算例目录中
cat > system/decomposeParDict <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }
numberOfSubdomains 16;
method simple;
coeffs { n (1 1 16); }      // 楔形网格沿 z 切；也可 (2 1 8)
EOF
blockMesh && cp 0/alpha.ethanol.orig 0/alpha.ethanol && setFields
decomposePar -force
mpirun -np 16 plasmaInterFoam -parallel > log.run 2>&1
reconstructPar
```
求解器已做并行改造并验证：4 核与串行结果一致（2 ms 后 α 最大差 3×10⁻⁴，气相格子数相同），快 3 倍。
16 核建议：一个算例用 16 核，或三个算例（0 kV / +2 / −2）各 5 核同时跑。

---

## 3. 求解器 plasmaInterFoam（v1912 interFoam + 唯象等离子体力）

未改动时与原版 interFoam 在 damBreak 上逐位一致。新增文件：
- `createPlasmaForce.H`：读 `constant/plasmaProperties`（global 对象，并行只读一份），预计算与时间无关的面系数，输出 `constant/polyMesh/C`（单元中心，供后处理）。
- `plasmaForce.H`：每步更新随机模态、在与针尖相连的气体中施加力；`UEqn.H`、`pEqn.H` 里在 `mixture.surfaceTensionForce()` 后加 `+ plasmaForcef`（面通量形式，与表面张力同一离散）。

控制方程（动量方程右端多两项）：

```
∇·U = 0
∂α/∂t + ∇·(αU) + ∇·[α(1−α)U_r] = 0             (MULES, cAlpha=1)
∂(ρU)/∂t + ∇·(ρUU) = −∇p_rgh − (g·x)∇ρ + ∇·[μ(∇U+∇Uᵀ)] + f_σ + f_EHD + f_pl
f_σ  = σ κ ∇α,  κ = −∇·(∇α/|∇α|)                (CSF，σ = 22.14 mN/m)
f_EHD = −F0 S(d) tanh(d/d0) (1−α) d̂ χc,  S(d) = ½[1 − tanh((d−R_E)/w_E)]
f_pl = [ P_up max(cosϑ,0)² + P_fl ξ(ϑ,t) W(ϑ) ] ∇α χc
ξ = (1/√N) Σ_{m=mMin..mMax} (a_m cos mϑ + b_m sin mϑ),  a_m,b_m 为 OU 过程（相关时间 τ）
W(ϑ) = ½[1 − tanh((ϑ − thetaCut)/dTheta)]       （扰动的角度窗，避开管口锚定区）
χc：从针尖 seedR 内的气体格子出发洪泛搜索得到的连通气体 + nLayer 层（并行下跨处理器同步）
```
d 为到放电点 x0（针尖）的距离，ϑ 为相对竖直轴的极角。
**符号**：+P∇α 把界面推向液相（向外）；−P∇α 是往里压（曾因此出错）。

`plasmaProperties` 关键字：`F0 RE wE d0 x0 axis Pfl Rfl tau mMin mMax seed Pup nUp seedR nLayer thetaCut dTheta`。

---

## 4. v1：论文几何（已完成）

- 几何：石英毛细管 ID 0.3 / OD 0.5 mm，He 3 mL/min（管内 0.707 m/s），乙醇 ρ 791、μ 1.17 mPa·s、σ 22.14 mN/m；He ρ 0.1625、μ 1.99e-5。
- 网格：轴对称 5° 楔形，r ≤ 5 mm，z ∈ [−3, 20] mm，近区 50 µm，20 722 格。接触角 45°（乙醇侧）。
- 初始：视频测得的残留气泡（球心 z=0.1 mm，R=1.54 mm，≈15 mm³）。
- 视频标定（mmc7）：148 px/mm（管外径 74 px）；时间基准由体积增长 0.0053 mm³/帧 ↔ 50 mm³/s 得 0.1 ms/帧；残留气泡顶 +1.57、底 −1.37 mm、宽 3.0 mm；负极脱离 7.7 mm³、残留 9.2 mm³；正极残留→断开约 52 ms。
- 最终参数：F0 = 40 ρ_l g（3.10e5 N/m³），R_E = 1.5 mm，w_E = 0.15 mm，d0 = 0.05 mm，P_up = 60 Pa，P_fl = 5 Pa（正）/ 8 Pa（负），m 4–24，τ 0.3 ms，seedR 0.8 mm。
- 结果：+2.0 kV 在 16.0–16.5 ms 断开，脱离 6.1 mm³、残留 9.8；−2.0 kV 11.5–12.0 ms，脱离 8.4、残留 7.2；0 kV 14.5–15 ms 整体脱离（单靠表面张力+浮力拴不住）。
- 调参路径（重要经验）：只有中心力时不颈缩 → 加顶部推力；推力符号写反 → 改 +P∇α；推力太大整体被推走 → F0 提高；脱离气泡被拉回 → 加连通掩码 χc；负极扰动 >25 Pa 整体被甩走 → 降到 8 Pa。
- 交付：`results/plasma_bubble_2p0kV.gif`、`results/OpenFOAM复刻_等离子体气泡_控制方程与过程.pptx`（6 页）、`results/零基础读懂_OpenFOAM复刻等离子体气泡.pdf`（16 页）。
- 局限：断开比实验快约 3 倍；负极褶皱远弱于视频；轴对称；未做网格收敛。

---

## 5. v2：用户新几何（**进行中，接着做这里**）

### 用户新条件（2026-11-27）
| 项目 | 设置 |
|---|---|
| 液体 | 无水乙醇 ρ 791、μ 1.17 mPa·s、σ 22.14 mN/m |
| 气体 | 氦气 ρ 0.1625、μ 1.99e-5 Pa·s |
| 电极 | 毛细管**内径 1 mm、外径 3 mm**，钨针与管口平齐，**环电极距针尖 10 mm** |
| 供气 | He 3 mL/min |

**待用户确认的两点**（已按默认值推进）：
1. 用户写“钨针半径 0.5 mm”，但这样针会填满内径 1 mm 的管子，气体无路可走。默认按**针直径 0.5 mm（半径 0.25 mm）**，氦气从环缝 0.25 < r < 0.5 mm 流出。
2. 用户表里“管内平均速度 0.707 m/s”是旧几何的数；新几何环缝面积 0.589 mm²，3 mL/min 对应 **0.085 m/s**（已采用）。
3. 环电极 10 mm 在唯象模型里不直接出现；等离子体参数仍沿用 v1 标定值（无新视频可标定），应向用户说明。

### 用户要求修正
1. **颈缩脱离后的气泡应更接近球形**，而不是扁平（v1 中脱离气泡上浮时偏扁）。思路：降低顶部推力 P_up，让断开主要由浮力驱动、上浮速度低一些；必要时网格加密到 25 µm；检查脱离瞬间是否仍有推力作用。新增的 `metrics.py` 会输出脱离气泡高宽比 `AR`（=1 为圆）。
2. **负极性褶皱不明显** → 用高阶模态（m 12–40）、更大幅值（≈40 Pa），并用 `thetaCut = 110°` 把扰动挡在管口锚定区之外（v1 中扰动松动锚定导致整体被甩走）。
3. **GIF 质量更高**：更高 dpi、更密的帧（0.25 ms）、更细的栅格；版式按第 0 节模板。

### 生成器
`case/make_case_v2.py`（几何参数 R_N=0.25、R_IN=0.5、R_OUT=1.5 mm；块网格：针和管壁块挖空；patch：inlet（环缝底面）、needle、capillary、bottom、outer、top、front/back wedge）。
50 µm 网格 36 764 格；计算域 r ≤ 8 mm（4.5 mm 内均匀），z ∈ [−3, 22] mm。初始气泡：`--Rb0` 半径、球心 `--zb0`，并把环缝填满气体。
示例：
```bash
python3 case/make_case_v2.py <dir> --mode pos --dx 50e-6 --kappa 40 --RE 2.0e-3 --wE 0.15e-3 \
   --Rb0 2.0e-3 --zb0 0 --Ppos 5 --Pup 30 --tend 0.06 --dtw 0.0005
python3 case/make_case_v2.py <dir> --mode neg ... --Pneg 40 --mMin 12 --mMax 40 --thetaCut 110 --dTheta 10
python3 case/make_case_v2.py <dir> --mode off ...     # 0 kV 对照
```

### 已跑的调参（云端 4 核，被用户叫停时的状态）
| 算例 | 设置 | 到达时间 | 观察 |
|---|---|---|---|
| v2c_A | R_E=Rb0=2.0 mm，P_up=30 Pa，P_fl=5 | 23 ms | 附着体积约 32 mm³，顶部在 3–4.2 mm 振荡，**尚未断开** |
| v2c_B | 同上，P_up=15 Pa | 23 ms | 同样未断开，顶部 2.5–4.2 mm |
| v2c_C | R_E=Rb0=2.5 mm，P_up=30 | 19.5 ms | 附着约 64 mm³，未断开 |
| v2n_40 | 负极，R_E=2.0，P_up=30，P_fl=40，m 12–40，thetaCut 110° | 13.8 ms | **仍附着在管口**（未被甩走），褶皱情况未出图检查 |

（metrics 中“底 −2.91 mm”是环缝里的气体被计入，非真实气泡底，可忽略。）

### 下一步建议
1. 用 16 核把上面 4 个算例各跑到 60–100 ms，看是否断开、脱离气泡 AR。若 60 ms 仍不断开：增大 P_up（40–60）或减小 R_E，使断开发生；同时比较 AR，选 AR 最接近 1 的一组。
2. 负极：出图检查 v2n_40 的褶皱；若仍不明显，P_fl 提到 60–80 Pa（保持 thetaCut），或把 τ 缩短到 0.1 ms。
3. 定参后跑最终三组（0 kV / +2.0 kV / −2.0 kV），网格建议 25 µm（需在 make_case_v2 中用 `--dx 25e-6`；单元约 15 万，16 核可承受）。
4. 出 GIF：`post/render_gif.py OUT.gif "dir_off:0 kV" "dir_pos:+2.0 kV" "dir_neg:−2.0 kV" --tmax 60 --dt 0.25`（可把 figsize/dpi 调高；需设置环境变量 `RUNS` 或改脚本里的路径）。
5. 更新 PPT/PDF 中的工况表为新几何。

---

## 6. 后处理脚本
- `post/ofio.py`：读 ASCII 场（`constant/polyMesh/C` 为单元中心）。
- `post/metrics.py CASE...`：每 0.5 ms 统计附着气泡顶/底/宽/体积、脱离气泡体积/位置/高宽比，写 `metrics.json`。
- `post/render_gif.py`：模板版式多栏 GIF；`--test T` 出单帧 PNG。
- `slides/make_figs.py`、`slides/build_deck.js`：PPT 图与生成（需 `pptxgenjs`）。
- `guide/`：零基础 PDF（HTML → Chromium 打印）。
- 注意：脚本中的数据路径默认指向云端 scratchpad（`/tmp/claude-0/...`），本地需改为算例所在目录。

---

## 7. 甲醇子项目（Gerris，已完成）要点
- 问题：甲醇中针–环电极 2.6 kV，针尖气相来源与起始判据。
- 结论：针尖场强 25 MV/m，液相直接电离不够 → 先需气相；焦耳沸腾判据 σe·V²/(8kΔT) ≥ 1，2.6 kV 临界电导率 9.4 µS/m；气核内放电判据 ΔV(d) ≥ V_Paschen(pd)；放电关闭时蒸气泡 0.9 ms 内冷凝塌缩，放电开启时气泡留存并积累不可凝气。
- 详见 `plasma_methanol_cfd/NOTES.md`；Gerris 需自行编译并打两个补丁（`case/*.patch`）。
