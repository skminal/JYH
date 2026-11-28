// Combined 6-slide deck: the two GIFs + their governing equations
// node build_deck.js  ->  制氢模拟/综合汇报/数值模拟GIF与控制方程_6页.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SK = "/root/.claude/skills/synced/046dc85a-87ec-41c2-aa9d-421c406ade6e_235fe0d1-3a13-48a8-8d0c-12692f04ee5a/pptx";
const { applyTheme } = require(SK + "/scripts/apply_theme.js");

const ROOT = path.join(__dirname, "..");
const F_M = path.join(__dirname, "fig");
const F_O = path.join(ROOT, "plasma_bubble_openfoam", "slides", "fig");
const GIF_M = path.join(ROOT, "plasma_methanol_cfd", "results", "needle_2p6kV_minimal.gif");
const GIF_O = path.join(ROOT, "plasma_bubble_openfoam", "results", "plasma_bubble_2p0kV.gif");

function size(p) {
  const b = fs.readFileSync(p);
  if (b.slice(0, 3).toString() === "GIF") return { w: b.readUInt16LE(6), h: b.readUInt16LE(8) };
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const THEME = {
  name: "Hydrogen plasma simulations",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A1A19", lt1: "FFFFFF", dk2: "55534E", lt2: "F3F2EE",
    accent1: "0B3D7A", accent2: "EB6C15", accent3: "1BAF7C", accent4: "4A42C6",
    accent5: "C2372C", accent6: "8F8D87", hlink: "2A78D6", folHlink: "4A42C6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "等离子体强化制氢：数值模拟 GIF 与控制方程";
const C = pres.SchemeColor;
const Wd = 13.33, M = 0.5;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.28, w: Wd - 2 * M, h: 0.8,
        fontSize: 26, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "等离子体强化制氢 · 数值模拟（Gerris / OpenFOAM v1912）", options: { x: M, y: 7.08, w: 9, h: 0.28, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: Wd - M - 0.6, y: 7.08, w: 0.6, h: 0.28, fontSize: 10, color: C.text2, align: "right" },
});

function slide(title) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", align: "left" }, o));
}
function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap === undefined ? 5 : o.gap } }));
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 13, color: C.text1, valign: "top", align: "left" }, o));
}
function card(s, x, y, w, h, name) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: name });
}
// place an image by width (capped by height); returns the size used
function img(s, p, x, y, w, alt, hmax) {
  const sz = size(p);
  let h = w * sz.h / sz.w;
  if (hmax && h > hmax) { w = w * hmax / h; h = hmax; }
  s.addImage({ path: p, x, y, w, h, altText: alt });
  return { w, h };
}
function table(s, rows, o) {
  const fs_ = o.fs || 11.5;
  const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 }, fontSize: fs_, color: C.text1 } });
  const cell = (t, b) => ({ text: t, options: { fontSize: fs_, color: C.text1, bold: !!b } });
  const data = [rows[0].map(hdr)].concat(rows.slice(1).map((r) => r.map((t, i) => cell(t, i === 0))));
  s.addTable(data, Object.assign({ border: { type: "solid", pt: 0.5, color: "D9D7D0" }, fontFace: THEME.bodyFontFace,
    margin: [0.035, 0.07, 0.035, 0.07], valign: "middle" }, o));
}
const head = (s, t, y, color) => txt(s, t, { x: M, y, w: 8.6, h: 0.32, fontSize: 14.5, bold: true, color: color || C.accent1 });

// ------------------------------------------------------------ 1 甲醇模型方程
{
  const s = slide("模型一：甲醇针电极放电（2.6 kV）的控制方程");
  const wl = 6.55;
  txt(s, "① 两相流：VOF + 表面张力 + 热浮力（相变、放电产气为体积源）", { x: M, y: 1.15, w: wl, h: 0.32, fontSize: 13.5, bold: true, color: C.accent1 });
  let a = img(s, path.join(F_M, "m_fluid.png"), M + 0.05, 1.55, wl - 0.1, "两相流控制方程", 1.85);
  let y = 1.55 + a.h + 0.25;
  txt(s, "② 电场：欧姆传导 + 电荷弛豫，放电区电导率随场强增大", { x: M, y, w: wl, h: 0.32, fontSize: 13.5, bold: true, color: C.accent1 });
  img(s, path.join(F_M, "m_elec.png"), M + 0.05, y + 0.4, wl - 0.6, "电场与电荷方程", 6.95 - (y + 0.4));
  const x2 = 7.35, w2 = Wd - M - x2;
  txt(s, "③ 能量（焦耳热）、蒸发/冷凝、放电产气", { x: x2, y: 1.15, w: w2, h: 0.32, fontSize: 13.5, bold: true, color: C.accent1 });
  a = img(s, path.join(F_M, "m_heat.png"), x2 + 0.05, 1.55, w2 - 0.1, "能量、相变与放电产气方程", 2.3);
  const yc = 1.55 + a.h + 0.25;
  card(s, x2, yc, w2, 6.95 - yc, "params");
  txt(s, "工况与参数", { x: x2 + 0.2, y: yc + 0.1, w: w2 - 0.4, h: 0.3, fontSize: 13.5, bold: true });
  bullets(s, [
    "钨针尖半径 0.04 mm、锥角 15°；环电极距针尖 30 mm；DC 2.6 kV；轴对称，最细网格 12.6 µm",
    "甲醇 25 °C：ρ 786.6 kg/m³，μ 0.544 mPa·s，σ 22.07 mN/m，εr 32.66，cp 2530 J/(kg·K)，k 0.20 W/(m·K)，Tsat 64.7 °C，hfg 1.10 MJ/kg",
    "σe = K_l = 44 µS/m（假定）；成核：针面 80 µm 内 θ ≥ 1.1",
    "放电：Kp = 0（关）/ 0.1 K_l（开）；放电功率 30% 用于甲醇分解；τr = 5Δt",
  ], { x: x2 + 0.2, y: yc + 0.45, w: w2 - 0.4, h: 6.85 - (yc + 0.45), fontSize: 11.5, gap: 4 });
}

// ------------------------------------------------------------ 2 甲醇 GIF
{
  const s = slide("模型一结果：针尖气相先由焦耳热产生，放电让气泡留得住");
  const g = img(s, GIF_M, M, 1.12, 8.2, "甲醇 2.6 kV 放电关闭与开启对照动图", 4.62);
  const yc = 1.12 + g.h + 0.06;
  txt(s, "(a) 放电关闭　(b) 放电开启（Kp = 0.1 σe）。底色：液体温升 θ；深蓝：气相；紫：放电功率密度。",
    { x: M, y: yc, w: g.w, h: 0.28, fontSize: 10.5, color: C.text2 });
  bullets(s, [
    "放电关闭：蒸气泡长到等效半径 52 µm 后被冷凝，成核后 0.92 ms 塌缩，随后再次成核",
    "放电开启：气泡持续增大到 4.8×10⁻³ mm³，并积累不可凝气（H₂、CO）2.7×10⁻⁴ mm³",
  ], { x: M, y: yc + 0.36, w: g.w, h: 0.75, fontSize: 11.5, gap: 2 });
  const x2 = 8.95, w2 = Wd - M - x2;
  table(s, [
    ["量", "数值"],
    ["针尖场强", "25 MV/m"],
    ["全针电流 / 焦耳功率", "0.59 mA / 1.54 W"],
    ["焦耳判据 θ∞", "4.7（临界 σe 9.4 µS/m）"],
    ["绝热针到沸点", "28.5 ms"],
    ["成核时刻", "34.94 ms"],
    ["0.1 mm 气核两端电压", "0.75 kV"],
  ], { x: x2, y: 1.12, w: w2, colW: [1.75, w2 - 1.75], fs: 11 });
  txt(s, "起始判据", { x: x2, y: 4.05, w: w2, h: 0.3, fontSize: 13.5, bold: true, color: C.accent1 });
  const c = img(s, path.join(F_M, "m_crit.png"), x2, 4.4, w2 - 0.2, "焦耳沸腾判据与 Paschen 击穿判据", 1.35);
  txt(s, "① 焦耳热能否把针尖液体烧开（导热极限）\n② 气核两端电压能否击穿气隙", { x: x2, y: 4.4 + c.h + 0.1, w: w2, h: 0.55, fontSize: 11, color: C.text2 });
  txt(s, "限制：等离子体电导率是假定的闭合；两相阶段 1.5–2 ms 后数值发散。", { x: x2, y: 6.5, w: w2, h: 0.5, fontSize: 10.5, color: C.accent5 });
}

// ------------------------------------------------------------ 3 OpenFOAM 两相流 + 表面张力
{
  const s = slide("模型二：乙醇中氦等离子体气泡（OpenFOAM）——两相流与表面张力");
  head(s, "连续性、VOF 相方程（MULES 界面压缩）与混合物性", 1.15);
  let a = img(s, path.join(F_O, "eq_vof.png"), M + 0.1, 1.55, 6.6, "连续性与相方程", 1.55);
  let y = 1.55 + a.h + 0.12;
  head(s, "动量方程（p_rgh 形式）", y);
  a = img(s, path.join(F_O, "eq_mom.png"), M + 0.1, y + 0.4, 8.2, "动量方程", 0.95);
  y = y + 0.4 + a.h + 0.12;
  head(s, "表面张力（CSF）、Laplace 压差、接触角、毛细时间步", y);
  img(s, path.join(F_O, "eq_sigma.png"), M + 0.1, y + 0.4, 6.6, "表面张力相关方程", 6.95 - (y + 0.4));
  const x2 = 9.15, w2 = Wd - M - x2;
  card(s, x2, 1.15, w2, 5.8, "params2");
  txt(s, "工况（论文 Wang, Zhang 等 2024）", { x: x2 + 0.2, y: 1.25, w: w2 - 0.4, h: 0.32, fontSize: 14, bold: true });
  bullets(s, [
    "乙醇：ρl 791 kg/m³，μl 1.17 mPa·s，σ 22.14 mN/m",
    "氦气：ρg 0.1625 kg/m³，μg 1.99×10⁻⁵ Pa·s",
    "石英毛细管 ID 0.3 / OD 0.5 mm，He 3 mL/min → 0.707 m/s",
    "轴对称 5° 楔形，r ≤ 5 mm，z ∈ [−3, 20] mm，近区 50 µm，20 722 格",
    "接触角 45°；初始为视频测得的残留气泡（R = 1.54 mm，15 mm³）",
    "R = 1.5 mm 气泡 Laplace 压差 ≈ 29 Pa，与等离子体界面力同量级：表面张力决定颈缩与断开",
  ], { x: x2 + 0.2, y: 1.7, w: w2 - 0.4, h: 5.1, fontSize: 12, gap: 6 });
}

// ------------------------------------------------------------ 4 等离子体闭合
{
  const s = slide("模型二：等离子体作用的唯象闭合（plasmaInterFoam）");
  txt(s, "依据论文 Fig. 1I：F_EHD 把气泡留在管口，顶部受向外的等离子体力；负极性界面扰动更强（Fig. 4）",
    { x: M, y: 1.12, w: Wd - 2 * M, h: 0.35, fontSize: 13, color: C.text2 });
  const a = img(s, path.join(F_O, "eq_ehd.png"), M + 0.1, 1.6, 8.3, "EHD 力、界面应力与随机扰动", 2.4);
  const b = img(s, path.join(F_O, "eq_balance.png"), M + 0.1, 1.6 + a.h + 0.15, 5.6, "力平衡与 Bond 数", 1.0);
  bullets(s, [
    "f_EHD：指向针尖 x0 的中心力，只作用于气相（1−α），R_E 内恒定、外侧衰减；旋度为零，不在气内驱动伪环流",
    "f_pl：以 ∇α 作用在界面上；P_up 项为顶部离子风推力，P_fl 项为随机扰动（OU 模态）",
    "χc：只对与针尖连通的气体施加（每步洪泛搜索）；脱离的气泡只受浮力和表面张力",
  ], { x: M, y: 1.6 + a.h + 0.15 + b.h + 0.18, w: 8.4, h: 1.7, fontSize: 12, gap: 4 });
  const x2 = 9.15, w2 = Wd - M - x2;
  table(s, [
    ["参数", "+2.0 kV", "−2.0 kV"],
    ["F0 / (ρl g)", "40", "40"],
    ["R_E / w_E (mm)", "1.5 / 0.15", "1.5 / 0.15"],
    ["P_up (Pa)", "60", "60"],
    ["P_fl (Pa)", "5", "8"],
    ["模态 m / τ", "4–24 / 0.3 ms", "4–24 / 0.3 ms"],
  ], { x: x2, y: 1.6, w: w2, colW: [1.5, (w2 - 1.5) / 2, (w2 - 1.5) / 2], fs: 12 });
  card(s, x2, 4.35, w2, 2.6, "caveat");
  txt(s, "参数由实验视频（mmc7）标定：残留气泡形状、颈缩位置、脱离体积；不是由放电物理推出。极性差别只体现在 P_fl。",
    { x: x2 + 0.18, y: 4.47, w: w2 - 0.36, h: 2.4, fontSize: 12.5, color: C.accent5 });
}

// ------------------------------------------------------------ 5 OpenFOAM GIF
{
  const s = slide("模型二结果：两种极性均重现“残留→颈缩→脱离”");
  const g = img(s, GIF_O, M, 1.1, 8.0, "0 kV、+2.0 kV、−2.0 kV 气泡演化动图", 5.5);
  txt(s, "0–60 ms，每帧 0.5 ms。深蓝：氦气；虚线：管口。", { x: M, y: 1.1 + g.h + 0.06, w: g.w, h: 0.3, fontSize: 10.5, color: C.text2 });
  const x2 = 8.85, w2 = Wd - M - x2;
  table(s, [
    ["", "断开", "脱离", "残留"],
    ["+2.0 kV 模拟", "16.0–16.5 ms", "6.1 mm³", "9.8 mm³"],
    ["−2.0 kV 模拟", "11.5–12.0 ms", "8.4 mm³", "7.2 mm³"],
    ["−2.0 kV 实验", "—", "7.7 mm³", "9.2 mm³"],
    ["0 kV 模拟", "14.5–15 ms", "15.8 mm³（整体）", "≈0"],
  ], { x: x2, y: 1.15, w: w2, colW: [1.25, 1.1, 1.0, w2 - 3.35], fs: 10.5 });
  bullets(s, [
    "0 kV：只有表面张力和浮力，15 mm³ 气泡 15 ms 内整体脱离——巨型气泡必须有 F_EHD",
    "±2.0 kV：下部被拴在管口，顶部被推高拉长，在保持区边缘（z ≈ 1.5–2 mm）颈缩断开",
    "负极性更早断开、脱离更多，方向与实验一致",
  ], { x: x2, y: 3.6, w: w2, h: 2.4, fontSize: 12, gap: 6 });
  txt(s, "限制：断开比实验快约 3 倍；负极褶皱弱于视频；轴对称；50 µm 网格未做收敛。", { x: x2, y: 6.2, w: w2, h: 0.75, fontSize: 11, color: C.accent5 });
}

// ------------------------------------------------------------ 6 对照与结论
{
  const s = slide("两个模型的方程对照与结论");
  table(s, [
    ["方程 / 项", "模型一：甲醇放电（Gerris）", "模型二：等离子体气泡（OpenFOAM）"],
    ["质量守恒", "∇·u = S_th + S_d（蒸发、冷凝、放电产气）", "∇·U = 0（He 由进口流入）"],
    ["界面捕捉", "VOF（几何重构），四叉树自适应", "VOF + MULES 界面压缩，固定楔形网格"],
    ["表面张力", "CSF，σ = 22.07 mN/m（甲醇）", "CSF，σ = 22.14 mN/m（乙醇）；接触角 45°"],
    ["重力 / 浮力", "ρg + 液相热浮力（Boussinesq）", "−(g·x)∇ρ（p_rgh 形式）"],
    ["电场", "∇·(ε∇φ) = −ρe，欧姆电荷弛豫", "不求解电场"],
    ["能量 / 相变", "焦耳热 K_l|E|²，界面蒸发/冷凝", "等温，无相变"],
    ["放电作用", "气相电导率 K_g(E)，放电功率 → 产气", "唯象力：F_EHD 拉住、P_up 推顶、P_fl 抖动"],
    ["物理问题", "气相从哪来？起始判据", "巨型气泡的保持、颈缩与脱离"],
  ], { x: M, y: 1.15, w: Wd - 2 * M, colW: [1.6, 5.35, Wd - 2 * M - 6.95], fs: 11.5 });
  const y0 = 5.05, cw = (Wd - 2 * M - 0.3) / 2;
  card(s, M, y0, cw, 1.9, "c1");
  txt(s, "结论", { x: M + 0.2, y: y0 + 0.1, w: cw - 0.4, h: 0.3, fontSize: 13.5, bold: true, color: C.accent3 });
  bullets(s, [
    "针尖第一口气由焦耳热（或已有气核）产生，放电让气泡留存并产生 H₂/CO",
    "表面张力 + 浮力拴不住巨型气泡；EHD 力把气泡拴在管口，表面张力决定颈缩与断开",
  ], { x: M + 0.2, y: y0 + 0.45, w: cw - 0.4, h: 1.4, fontSize: 12, gap: 4 });
  const x2 = M + cw + 0.3;
  card(s, x2, y0, cw, 1.9, "c2");
  txt(s, "下一步", { x: x2 + 0.2, y: y0 + 0.1, w: cw - 0.4, h: 0.3, fontSize: 13.5, bold: true, color: C.accent2 });
  bullets(s, [
    "实测甲醇电导率与放电电流，标定放电闭合参数",
    "新几何（ID 1 / OD 3 mm、环距 10 mm）在 16 核上重算：脱离气泡更圆、负极褶皱更明显",
  ], { x: x2 + 0.2, y: y0 + 0.45, w: cw - 0.4, h: 1.4, fontSize: 12, gap: 4 });
}

(async () => {
  const out = path.join(__dirname, "数值模拟GIF与控制方程_6页.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
