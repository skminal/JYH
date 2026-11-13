// Group-meeting deck: where does the gas at the needle tip come from?
// node build_deck.js   ->  ../results/甲醇针尖气相来源_组会汇报.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SK = "/root/.claude/skills/synced/046dc85a-87ec-41c2-aa9d-421c406ade6e_235fe0d1-3a13-48a8-8d0c-12692f04ee5a/pptx";
const { applyTheme } = require(SK + "/scripts/apply_theme.js");

const RES = path.join(__dirname, "..", "results");
const K = JSON.parse(fs.readFileSync(path.join(RES, "key_numbers.json")));
const S = JSON.parse(fs.readFileSync(path.join(RES, "summary.json")));
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_text.json")));

const THEME = {
  name: "Needle discharge",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: "1A1A19", lt1: "FFFFFF", dk2: "55534E", lt2: "F3F2EE",
    accent1: "EB6C15", accent2: "2A78D6", accent3: "1BAF7C", accent4: "4A42C6",
    accent5: "C2372C", accent6: "8F8D87", hlink: "2A78D6", folHlink: "4A42C6",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "甲醇中针电极放电：针尖气相来源与起始判据";
const C = pres.SchemeColor;
const Wd = 13.33, M = 0.6;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.38, w: Wd - 2 * M, h: 0.9,
        fontSize: 28, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "甲醇针电极放电 · 2.6 kV 数值实验", options: { x: M, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: Wd - M - 0.6, y: 7.02, w: 0.6, h: 0.3, fontSize: 10, color: C.text2, align: "right" },
});

let n = 0;
function slide(title) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(title, { placeholder: "title" });
  n += 1;
  return s;
}
function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", align: "left" }, o));
}
function card(s, x, y, w, h, name) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2, width: 0 }, objectName: name });
}
function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: o.gap === undefined ? 7 : o.gap } }));
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text1, valign: "top", align: "left" }, o));
}
const f = (v, d = 1) => Number(v).toFixed(d);

// ------------------------------------------------------------------ 1 结论
{
  const s = slide(D.s1.title);
  txt(s, D.s1.lead, { x: M, y: 1.35, w: Wd - 2 * M, h: 0.75, fontSize: 16, color: C.text2 });
  const cw = 3.84, gap = 0.3, y0 = 2.3, h = 2.75;
  D.s1.cards.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, y0, cw, h, "card" + i);
    txt(s, c.big, { x: x + 0.3, y: y0 + 0.22, w: cw - 0.6, h: 0.9, fontSize: 40, bold: true, color: [C.accent1, C.accent2, C.accent3][i], valign: "middle" });
    txt(s, c.label, { x: x + 0.3, y: y0 + 1.15, w: cw - 0.6, h: 0.4, fontSize: 15, bold: true });
    txt(s, c.body, { x: x + 0.3, y: y0 + 1.58, w: cw - 0.6, h: 1.1, fontSize: 13, color: C.text2 });
  });
  txt(s, "回答", { x: M, y: 5.32, w: 1.0, h: 0.4, fontSize: 15, bold: true, color: C.accent1 });
  bullets(s, D.s1.answer, { x: M, y: 5.72, w: Wd - 2 * M, h: 1.2, fontSize: 14, gap: 4 });
}

// ------------------------------------------------------------------ 2 工况与模型
{
  const s = slide(D.s2.title);
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.text1, fill: { color: C.background2 }, fontSize: 12 } });
  const cell = (t, b) => ({ text: t, options: { fontSize: 11.5, color: C.text1, bold: !!b } });
  const rows = [[hdr("项目"), hdr("文献 Wang et al., Matter 2024"), hdr("本算例")]].concat(
    D.s2.table.map((r) => [cell(r[0], true), cell(r[1]), cell(r[2])]));
  s.addTable(rows, { x: M, y: 1.4, w: 7.55, colW: [1.25, 2.9, 3.4], border: { type: "solid", pt: 0.5, color: "D9D7D0" },
    fontFace: THEME.bodyFontFace, margin: [0.045, 0.08, 0.045, 0.08], valign: "middle", objectName: "conditions" });
  const x2 = 8.5, w2 = Wd - M - x2;
  txt(s, "模型（轴对称，Gerris + electrohydro）", { x: x2, y: 1.4, w: w2, h: 0.35, fontSize: 15, bold: true });
  bullets(s, D.s2.model, { x: x2, y: 1.82, w: w2, h: 2.55, fontSize: 12.5, gap: 4 });
  txt(s, "假设与未包含项", { x: x2, y: 4.45, w: w2, h: 0.35, fontSize: 15, bold: true, color: C.accent5 });
  bullets(s, D.s2.assume, { x: x2, y: 4.87, w: w2, h: 2.0, fontSize: 12.5, gap: 4 });
}

// ------------------------------------------------------------------ 3 焦耳热
{
  const s = slide(D.s3.title);
  s.addImage({ path: path.join(RES, "fig_joule.png"), x: M, y: 1.4, w: 8.1, h: 8.1 * 672 / 1760, altText: "焦耳升温曲线与沸腾判据图" });
  const x2 = 9.0, w2 = Wd - M - x2;
  card(s, x2, 1.4, w2, 1.5, "crit1");
  txt(s, "判据 ①", { x: x2 + 0.25, y: 1.5, w: w2 - 0.5, h: 0.3, fontSize: 12, bold: true, color: C.accent1 });
  txt(s, "σe·V² / (8 k ΔT) ≥ 1", { x: x2 + 0.25, y: 1.82, w: w2 - 0.5, h: 0.5, fontSize: 22, bold: true });
  txt(s, "k 导热系数，ΔT = Tsat − T0；与电极形状无关", { x: x2 + 0.25, y: 2.38, w: w2 - 0.5, h: 0.4, fontSize: 11.5, color: C.text2 });
  bullets(s, D.s3.points, { x: x2, y: 3.1, w: w2, h: 3.7, fontSize: 13, gap: 6 });
  txt(s, D.s3.note, { x: M, y: 4.65, w: 8.1, h: 2.2, fontSize: 13, color: C.text2 });
}

// ------------------------------------------------------------------ 4 放电
{
  const s = slide(D.s4.title);
  s.addImage({ path: path.join(RES, "fig_discharge.png"), x: M, y: 1.4, w: 8.1, h: 8.1 * 672 / 1760, altText: "针尖轴线场强与气核击穿判据图" });
  const x2 = 9.0, w2 = Wd - M - x2;
  card(s, x2, 1.4, w2, 1.5, "crit2");
  txt(s, "判据 ②", { x: x2 + 0.25, y: 1.5, w: w2 - 0.5, h: 0.3, fontSize: 12, bold: true, color: C.accent2 });
  txt(s, "ΔV(d) ≥ V_Paschen(p·d)", { x: x2 + 0.25, y: 1.82, w: w2 - 0.5, h: 0.5, fontSize: 20, bold: true });
  txt(s, "气核上的电压 ≥ 该尺寸气隙的击穿电压（Townsend 自持）", { x: x2 + 0.25, y: 2.38, w: w2 - 0.5, h: 0.4, fontSize: 11.5, color: C.text2 });
  bullets(s, D.s4.points, { x: x2, y: 3.1, w: w2, h: 3.7, fontSize: 13, gap: 6 });
  txt(s, D.s4.note, { x: M, y: 4.65, w: 8.1, h: 2.2, fontSize: 13, color: C.text2 });
}

// ------------------------------------------------------------------ 5 数值实验
{
  const s = slide(D.s5.title);
  const gw = 7.9, gh = gw * 720 / 1280;
  s.addImage({ path: path.join(RES, D.s5.gif), x: M, y: 1.4, w: gw, h: gh, altText: "放电关闭与开启两种情况下针尖气相演化的动画" });
  txt(s, "动图（放映时播放）：左 放电关闭，右 放电开启；右侧为针尖温升、气相体积、功率曲线", { x: M, y: 1.4 + gh + 0.08, w: gw, h: 0.3, fontSize: 10.5, color: C.text2 });
  const x2 = 8.8, w2 = Wd - M - x2;
  const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 }, fontSize: 11.5, color: C.text1 } });
  const cell = (t, b) => ({ text: t, options: { fontSize: 11.5, color: C.text1, bold: !!b } });
  const rows = [D.s5.table[0].map(hdr)].concat(D.s5.table.slice(1).map((r) => r.map((t, i) => cell(t, i === 0))));
  s.addTable(rows, { x: x2, y: 1.4, w: w2, colW: [1.33, 0.87, 0.87, 0.86], border: { type: "solid", pt: 0.5, color: "D9D7D0" },
    fontFace: THEME.bodyFontFace, margin: [0.04, 0.06, 0.04, 0.06], valign: "middle", objectName: "variants" });
  bullets(s, D.s5.points, { x: x2, y: 4.3, w: w2, h: 2.6, fontSize: 12.5, gap: 5 });
}

// ------------------------------------------------------------------ 6 结论与下一步
{
  const s = slide(D.s6.title);
  const cw = (Wd - 2 * M - 0.4) / 2;
  txt(s, "这次算出来的", { x: M, y: 1.4, w: cw, h: 0.35, fontSize: 16, bold: true, color: C.accent3 });
  bullets(s, D.s6.found, { x: M, y: 1.85, w: cw, h: 2.6, fontSize: 13.5, gap: 6 });
  txt(s, "还不能下结论的", { x: M, y: 4.5, w: cw, h: 0.35, fontSize: 16, bold: true, color: C.accent5 });
  bullets(s, D.s6.open, { x: M, y: 4.95, w: cw, h: 1.95, fontSize: 13.5, gap: 6 });
  const x2 = M + cw + 0.4;
  card(s, x2, 1.4, cw, 5.45, "next");
  txt(s, "下一步：用实验把判据钉死", { x: x2 + 0.3, y: 1.55, w: cw - 0.6, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
  const runs = [];
  D.s6.next.forEach((it, i) => {
    runs.push({ text: (i + 1) + "  " + it[0], options: { bold: true, breakLine: true, fontSize: 13.5, paraSpaceAfter: 1 } });
    runs.push({ text: it[1], options: { breakLine: i < D.s6.next.length - 1, fontSize: 12.5, color: C.text2, paraSpaceAfter: 9 } });
  });
  s.addText(runs, { isTextBox: true, margin: 0, x: x2 + 0.3, y: 2.0, w: cw - 0.6, h: 4.7, valign: "top", color: C.text1 });
}

// ------------------------------------------------------------------ 备份：验证
{
  const s = slide(D.b1.title);
  const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: C.background2 }, fontSize: 12, color: C.text1 } });
  const cell = (t, b) => ({ text: t, options: { fontSize: 12, color: C.text1, bold: !!b } });
  const rows = [D.b1.table[0].map(hdr)].concat(D.b1.table.slice(1).map((r) => r.map((t, i) => cell(t, i === 0))));
  s.addTable(rows, { x: M, y: 1.4, w: Wd - 2 * M, colW: [3.0, 5.0, 4.13], border: { type: "solid", pt: 0.5, color: "D9D7D0" },
    fontFace: THEME.bodyFontFace, margin: [0.06, 0.1, 0.06, 0.1], valign: "middle", objectName: "verification" });
  txt(s, "数值设置与已知问题", { x: M, y: 4.2, w: 6, h: 0.35, fontSize: 15, bold: true });
  bullets(s, D.b1.points, { x: M, y: 4.62, w: Wd - 2 * M, h: 2.3, fontSize: 12.5, gap: 4 });
}

(async () => {
  const out = path.join(RES, "甲醇针尖气相来源_组会汇报.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out, n, "slides");
})();
