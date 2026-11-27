// OpenFOAM reproduction of the 2.0 kV plasma-bubble experiment (Wang, Zhang et al., Matter 2024)
// node build_deck.js  ->  ../results/OpenFOAM复刻_等离子体气泡_控制方程与过程.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SK = "/root/.claude/skills/synced/046dc85a-87ec-41c2-aa9d-421c406ade6e_235fe0d1-3a13-48a8-8d0c-12692f04ee5a/pptx";
const { applyTheme } = require(SK + "/scripts/apply_theme.js");
const { imageSize } = (() => {
  // read PNG width/height from the IHDR chunk
  return {
    imageSize: (p) => {
      const b = fs.readFileSync(p);
      return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
    },
  };
})();

const FIG = path.join(__dirname, "fig");
const OUTDIR = path.join(__dirname, "..", "results");

const THEME = {
  name: "Plasma bubble OpenFOAM",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A1A19", lt1: "FFFFFF", dk2: "55534E", lt2: "F3F2EE",
    accent1: "0B3D7A", accent2: "EB6C15", accent3: "1BAF7C", accent4: "4A42C6",
    accent5: "C2372C", accent6: "8F8D87", hlink: "2A78D6", folHlink: "4A42C6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "OpenFOAM 复刻：乙醇中氦等离子体气泡（2.0 kV）";
const C = pres.SchemeColor;
const Wd = 13.33, M = 0.5;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.3, w: Wd - 2 * M, h: 0.8,
        fontSize: 26, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "OpenFOAM v1912 · plasmaInterFoam · 复刻 Wang, Zhang et al., Matter 7 (2024) 3024", options: { x: M, y: 7.08, w: 9, h: 0.28, fontSize: 10, color: C.text2, margin: 0 } } },
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
// image placed by width; returns its height
function img(s, name, x, y, w, alt, hmax) {
  const p = path.join(FIG, name);
  const sz = imageSize(p);
  let h = w * sz.h / sz.w;
  if (hmax && h > hmax) { w = w * hmax / h; h = hmax; }
  s.addImage({ path: p, x, y, w, h, altText: alt });
  return { w, h };
}
function table(s, rows, o) {
  const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 }, fontSize: o.fs || 11.5, color: C.text1 } });
  const cell = (t, b) => ({ text: t, options: { fontSize: o.fs || 11.5, color: C.text1, bold: !!b } });
  const data = [rows[0].map(hdr)].concat(rows.slice(1).map((r) => r.map((t, i) => cell(t, i === 0))));
  s.addTable(data, Object.assign({ border: { type: "solid", pt: 0.5, color: "D9D7D0" }, fontFace: THEME.bodyFontFace,
    margin: [0.035, 0.07, 0.035, 0.07], valign: "middle" }, o));
}

// ------------------------------------------------------------ 1 复刻对象与工况
{
  const s = slide("复刻对象：2.0 kV 正/负极性等离子体气泡的“残留→长大→颈缩→脱离”");
  const r = img(s, "video_frames.png", M, 1.25, 7.3, "实验视频 mmc7 正负极性四个阶段的截图", 4.35);
  txt(s, "实验视频 mmc7（10000 fps）。比例尺由毛细管外径 0.5 mm = 74 px 得到（148 px/mm）；虚线 = 管口/针尖（z = 0）。",
    { x: M, y: 1.25 + r.h + 0.08, w: 7.3, h: 0.5, fontSize: 11, color: C.text2 });
  bullets(s, [
    "时间基准：+2.0 kV 气泡体积每帧增 0.0053 mm³，与供气 3 mL/min（50 mm³/s）对应 0.1 ms/帧",
    "残留气泡：顶 +1.57 mm，底 −1.37 mm，宽 3.0 mm，体积 15.0 mm³（作为初始条件）",
    "+2.0 kV：表面光滑，顶部升至 ≈5 mm 后颈缩脱离，残留→脱离约 52 ms",
    "−2.0 kV：表面强烈褶皱；脱离 7.7 mm³，残留 9.2 mm³",
  ], { x: M, y: 6.05, w: 7.3, h: 1.0, fontSize: 12, gap: 2 });
  const x2 = 8.15, w2 = Wd - M - x2;
  table(s, [
    ["项目", "设置"],
    ["液体", "无水乙醇：ρl 791 kg/m³，μl 1.17 mPa·s，σ 22.14 mN/m"],
    ["气体", "氦气：ρg 0.1625 kg/m³，μg 1.99×10⁻⁵ Pa·s"],
    ["电极", "石英毛细管 ID 0.3 / OD 0.5 mm，钨针尖与管口平齐；环电极距针尖 30 mm"],
    ["供气", "He 3 mL/min → 管内平均速度 0.707 m/s"],
    ["电压", "DC ±2.0 kV（正/负极性）；对照 0 kV"],
    ["计算域", "轴对称楔形 5°，r ≤ 5 mm，z ∈ [−3, 20] mm"],
    ["网格", "20 722 单元；近区 Δx = 50 µm（管壁 2 格，内径 3 格）"],
    ["初始", "视频残留气泡：球心 z = 0.1 mm，R = 1.54 mm"],
    ["时长", "0–60 ms（0 kV 算到 80 ms）；输出间隔 0.5 ms"],
  ], { x: x2, y: 1.25, w: w2, colW: [0.95, w2 - 0.95], fs: 11.5 });
}

// ------------------------------------------------------------ 2 控制方程：两相流
{
  const s = slide("控制方程（一）：不可压两相流 + VOF 界面捕捉");
  txt(s, "连续性方程、相分数输运（MULES 界面压缩）与混合物性", { x: M, y: 1.2, w: 8, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  const a = img(s, "eq_vof.png", M + 0.1, 1.65, 7.4, "连续性方程与 VOF 相方程");
  const y2 = 1.65 + a.h + 0.3;
  txt(s, "动量方程（p_rgh 形式，三项界面/体积力单独列出）", { x: M, y: y2, w: 8, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  img(s, "eq_mom.png", M + 0.1, y2 + 0.45, 8.6, "两相动量方程");
  const x2 = 9.55, w2 = Wd - M - x2;
  card(s, x2, 1.2, w2, 5.7, "symbols");
  txt(s, "符号与取值", { x: x2 + 0.2, y: 1.3, w: w2 - 0.4, h: 0.35, fontSize: 14, bold: true });
  bullets(s, [
    "α：乙醇体积分数（液 = 1，气 = 0）",
    "U：混合速度；Ur：界面压缩速度，cα = 1",
    "ρl = 791，ρg = 0.1625 kg/m³（密度比 4870）",
    "μl = 1.17×10⁻³，μg = 1.99×10⁻⁵ Pa·s",
    "g = (0, 0, −9.81) m/s²",
    "f_σ：表面张力（第 3 页）",
    "f_EHD、f_pl：等离子体作用（第 4 页）",
    "求解：PIMPLE，3 次压力修正；MULES 两次校正；Euler 隐式；maxCo = maxAlphaCo = 0.4",
    "层流（Re = ρl U D/μl ≈ 140，按 0.2 m/s、1 mm）",
  ], { x: x2 + 0.2, y: 1.75, w: w2 - 0.4, h: 5.05, fontSize: 12.5, gap: 6 });
}

// ------------------------------------------------------------ 3 控制方程：表面张力
{
  const s = slide("控制方程（二）：表面张力（CSF 模型）与接触角");
  txt(s, "连续表面力、Laplace 压差、壁面接触角与毛细时间步约束", { x: M, y: 1.2, w: 8.5, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  const a = img(s, "eq_sigma.png", M + 0.1, 1.65, 7.2, "表面张力 CSF 公式、Laplace 压差、接触角与时间步约束", 3.0);
  const y2 = 1.65 + a.h + 0.25;
  txt(s, "面通量上的平衡力离散（表面张力与压力梯度在同一面上计算，抑制伪速度）", { x: M, y: y2, w: 8.6, h: 0.35, fontSize: 15, bold: true, color: C.accent1 });
  img(s, "eq_face.png", M + 0.1, y2 + 0.45, 7.9, "面通量离散与压力方程", 6.85 - (y2 + 0.45));
  const x2 = 9.15, w2 = Wd - M - x2;
  card(s, x2, 1.2, w2, 5.7, "sigma-notes");
  txt(s, "要点", { x: x2 + 0.2, y: 1.3, w: w2 - 0.4, h: 0.35, fontSize: 14, bold: true });
  bullets(s, [
    "σ = 22.14 mN/m（乙醇，原文给定），三个算例均开启",
    "κ 由 α 梯度直接计算；n̂ 指向液相，气泡 κ < 0，故 f_σ 压缩气泡、使气内压高于液压",
    "R = 1.5 mm 气泡 Laplace 压差 ≈ 29 Pa，与等离子体界面应力（5–60 Pa）同量级：表面张力决定颈缩与断开",
    "毛细管壁 constantAlphaContactAngle θw = 45°（经乙醇侧测量）",
    "毛细时间步上限 ≈ 1.9×10⁻⁵ s；实际平均 Δt = 8.9–9.9 µs（由 Co = 0.4 控制）",
    "0 kV 对照：只有表面张力 + 浮力，15 mm³ 气泡在 14.5–15 ms 整体脱离（Bo ≈ 3.2 > 1，浮力胜过表面张力）",
  ], { x: x2 + 0.2, y: 1.75, w: w2 - 0.4, h: 5.05, fontSize: 12.5, gap: 7 });
}

// ------------------------------------------------------------ 4 等离子体闭合
{
  const s = slide("控制方程（三）：等离子体作用的唯象闭合（自编 plasmaInterFoam）");
  txt(s, "依据原文 Fig. 1I 的受力：F_EHD 把气泡留在管口；顶部受向外的等离子体力；负极界面扰动更强（Fig. 4）",
    { x: M, y: 1.2, w: Wd - 2 * M, h: 0.35, fontSize: 14, color: C.text2 });
  const a = img(s, "eq_ehd.png", M + 0.1, 1.65, 8.3, "EHD 体积力、界面应力与随机扰动的公式", 2.4);
  const b = img(s, "eq_balance.png", M + 0.1, 1.65 + a.h + 0.15, 5.6, "力平衡与 Bond 数", 1.0);
  bullets(s, [
    "f_EHD：指向放电点 x0 的中心力，只作用于气相（1−α），R_E 内恒定、外侧 tanh 衰减；旋度为零，不在气内驱动伪环流",
    "f_pl：以 ∇α 加在界面上；第一项为顶部离子风推力（ϑ 为相对竖直轴的极角），第二项为随机扰动（Ornstein–Uhlenbeck 模态）",
    "χc：只对与针尖连通的气体（种子半径 0.8 mm，每步洪泛搜索 + 2 层单元）施加；脱离气泡只受浮力和表面张力",
  ], { x: M, y: 1.65 + a.h + 0.15 + b.h + 0.18, w: 8.4, h: 1.6, fontSize: 12, gap: 3 });
  const x2 = 9.15, w2 = Wd - M - x2;
  table(s, [
    ["参数", "+2.0 kV", "−2.0 kV"],
    ["F0 / (ρl g)", "40", "40"],
    ["F0 (N/m³)", "3.10×10⁵", "3.10×10⁵"],
    ["R_E / w_E (mm)", "1.5 / 0.15", "1.5 / 0.15"],
    ["d0 (mm)", "0.05", "0.05"],
    ["P_up (Pa)", "60", "60"],
    ["P_fl (Pa)", "5", "8"],
    ["模态 m", "4–24", "4–24"],
    ["τ (ms)", "0.3", "0.3"],
  ], { x: x2, y: 1.7, w: w2, colW: [1.55, (w2 - 1.55) / 2, (w2 - 1.55) / 2], fs: 12 });
  card(s, x2, 5.1, w2, 1.85, "caveat");
  txt(s, "这些参数是对实验视频标定的结果（残留形状、颈缩、脱离体积），不是由放电物理推出；极性差别只体现在 P_fl。",
    { x: x2 + 0.18, y: 5.22, w: w2 - 0.36, h: 1.65, fontSize: 12.5, color: C.accent5 });
}

// ------------------------------------------------------------ 5 实现与标定
{
  const s = slide("OpenFOAM 实现与标定过程");
  img(s, "geometry.png", M, 1.2, 6.2, "计算域、边界条件与近区网格示意", 5.75);
  const x2 = 6.95, w2 = Wd - M - x2;
  txt(s, "求解器", { x: x2, y: 1.2, w: w2, h: 0.32, fontSize: 14, bold: true, color: C.accent1 });
  bullets(s, [
    "Ubuntu OpenFOAM v1912 + 源码包头文件，复制 interFoam 为 plasmaInterFoam 编译",
    "未改动时与原版 interFoam 在 damBreak 上逐位一致（α 最大差 0.0）",
    "f_EHD、f_pl 写入面通量（与表面张力同一位置），见第 3 页离散式",
  ], { x: x2, y: 1.55, w: w2, h: 1.25, fontSize: 12, gap: 3 });
  txt(s, "标定路径（每步用视频指标判断）", { x: x2, y: 2.85, w: w2, h: 0.32, fontSize: 14, bold: true, color: C.accent1 });
  table(s, [
    ["设置", "现象", "改动"],
    ["只有中心力 F0 = 14.7 ρl g，R_E = 1.7 mm", "33 ms 内顶部停在 ≈1.6 mm", "R_E 改 1.5 mm；初始体积按视频 15 mm³"],
    ["同上，R_E = 1.5 mm", "80 ms 内顶部在 2.3–3.3 mm 振荡，不颈缩", "加顶部推力 P_up"],
    ["P_up 首版", "顶部反被压低", "界面应力符号改为 +P∇α"],
    ["P_up = 60–100 Pa", "整个气泡被推离管口", "F0 提高到 40 ρl g"],
    ["F0 = 25 ρl g，P_up = 40 Pa", "脱离后又被拉回合并", "加连通掩码 χc"],
    ["负极 P_fl = 25–40 Pa", "约 3 ms 起整体脱离", "P_fl 降至 8 Pa"],
  ], { x: x2, y: 3.22, w: w2, colW: [1.75, 2.0, w2 - 3.75], fs: 10.5 });
  txt(s, "单个算例：约 6 000–9 000 步，单核 16–27 min", { x: x2, y: 6.65, w: w2, h: 0.3, fontSize: 11, color: C.text2 });
}

// ------------------------------------------------------------ 6 结果
{
  const s = slide("结果：两种极性均重现“残留→颈缩→脱离”，脱离体积与实验同量级");
  img(s, "key_frames.png", M, 1.15, 6.7, "三种工况在五个时刻的气泡形态", 5.85);
  const x2 = 7.45, w2 = Wd - M - x2;
  table(s, [
    ["", "颈缩断开", "脱离体积", "残留体积"],
    ["+2.0 kV 模拟", "16.0–16.5 ms", "6.1 mm³", "9.8 mm³"],
    ["−2.0 kV 模拟", "11.5–12.0 ms", "8.4 mm³", "7.2 mm³"],
    ["−2.0 kV 实验", "—", "7.7 mm³", "9.2 mm³"],
    ["+2.0 kV 实验", "≈52 ms", "—", "—"],
    ["0 kV 模拟", "14.5–15 ms", "15.8 mm³（整体）", "≈0"],
  ], { x: x2, y: 1.2, w: w2, colW: [1.3, 1.45, 1.6, w2 - 4.35], fs: 11.5 });
  txt(s, "体会", { x: x2, y: 3.75, w: w2, h: 0.32, fontSize: 14, bold: true, color: C.accent1 });
  bullets(s, [
    "没有把气泡留在管口的力，15 mm³ 气泡在表面张力 + 浮力下 15 ms 内整体脱离；巨型气泡需要 F_EHD",
    "颈缩位置在保持区边缘（z ≈ 1.5–2 mm），与“断开发生在气泡体内”一致",
  ], { x: x2, y: 4.1, w: w2, h: 1.2, fontSize: 12, gap: 4 });
  txt(s, "局限", { x: x2, y: 5.3, w: w2, h: 0.32, fontSize: 14, bold: true, color: C.accent5 });
  bullets(s, [
    "断开比实验快约 3 倍（顶部推力偏强）",
    "负极 P_fl 只能用 8 Pa，表面褶皱远弱于视频",
    "轴对称，非轴对称褶皱无法表现；50 µm 网格未做收敛",
  ], { x: x2, y: 5.65, w: w2, h: 1.3, fontSize: 12, gap: 3 });
}

(async () => {
  fs.mkdirSync(OUTDIR, { recursive: true });
  const out = path.join(OUTDIR, "OpenFOAM复刻_等离子体气泡_控制方程与过程.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
