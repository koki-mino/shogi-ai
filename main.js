/* Shogi NN (TFJS) Demo main.js v3
   - まず tf.loadGraphModel() で「モデルが読めるか」だけを確定
   - その後に fetch(model.json) で signature を“表示用に”読む（失敗してもOK）
   - 入力は 2283 次元（NNUE“風”）を想定
*/

/* ------------------ TFJS Model (GraphModel) ------------------ */
const MODEL_URL = "./shogi_eval_wars_tfjs/model.json";

let tfModel = null;
let tfModelStatus = "未読込";

// ★デフォルト（あなたの model.json で確定している値）
let tfInputKey = "input_layer_5";        // signature.inputs のキー
let tfInputTensorName = "input_layer_5:0";
let tfOutputTensorName = "Identity:0";
let tfExpectedD = 2283;

function pickFirstKey(obj) {
  if (!obj) return null;
  const ks = Object.keys(obj);
  return ks.length ? ks[0] : null;
}

async function loadTfModel() {
  // 1) まず「モデルが読めるか」だけを試す（ここが本命）
  try {
    tfModelStatus = "読込中...";
    setStatus();

    tfModel = await tf.loadGraphModel(MODEL_URL, { fromTFHub: false });
    tfModelStatus = "読込完了";
  } catch (e) {
    console.error(e);
    tfModelStatus = "読込失敗（手書き評価にフォールバック）: " + (e?.message || e);
    tfModel = null;
    setStatus();
    return;
  }

  // 2) 次に signature を “表示用に” 読む（失敗してもモデルは使える）
  try {
    const meta = await (await fetch(MODEL_URL, { cache: "no-store" })).json();
    const sigIn = meta?.signature?.inputs || null;
    const sigOut = meta?.signature?.outputs || null;

    const inKey = pickFirstKey(sigIn);
    const outKey = pickFirstKey(sigOut);

    if (inKey && sigIn[inKey]) {
      tfInputKey = inKey;
      tfInputTensorName = sigIn[inKey].name || tfInputTensorName;
      const dim = sigIn[inKey]?.tensorShape?.dim || [];
      const d1 = dim.length >= 2 ? parseInt(dim[1].size, 10) : NaN;
      if (Number.isFinite(d1)) tfExpectedD = d1;
    }

    if (outKey && sigOut[outKey]) {
      tfOutputTensorName = sigOut[outKey].name || tfOutputTensorName;
    }
  } catch (e) {
    // ここで失敗してもOK（表示情報が取れないだけ）
    console.warn("signature fetch failed (ignored):", e);
  } finally {
    setStatus();
  }
}

window.addEventListener("load", loadTfModel);

/* ------------------ NNUE“風” encoding (D=2283) ------------------ */
const NUM_PTYPE = 14;
const NUM_SQ = 81;
const HAND_ORDER = ['FU','KY','KE','GI','KI','KA','HI'];
const D = 1 + (2 * NUM_PTYPE * NUM_SQ) + (2 * HAND_ORDER.length);

// python-shogi の piece_type(1..14) 相当を想定
const PTYPE_ID = {
  FU: 1, KY: 2, KE: 3, GI: 4, KI: 5, KA: 6, HI: 7,
  OU: 8, TO: 9, NY: 10, NK: 11, NG: 12, UM: 13, RY: 14,
};

// ★sq の並び（まずは仮で y*9+x）
function sqIndex(x, y) {
  return y * 9 + x;
}

function encodePositionForNNUE(board, hands, sideToMove) {
  const x = new Float32Array(D);

  // 手番（先手=1, 後手=0）
  x[0] = (sideToMove === 'S') ? 1.0 : 0.0;

  // 盤面 one-hot
  const base = 1;
  for (let y = 0; y < 9; y++) for (let x0 = 0; x0 < 9; x0++) {
    const p = board[y][x0];
    if (!p) continue;

    const color = (p.side === 'S') ? 0 : 1; // 先手=0, 後手=1
    const pt = PTYPE_ID[p.type] || 0;       // 1..14
    if (!pt) continue;

    const sq = sqIndex(x0, y); // 0..80
    const pidx = pt - 1;       // 0..13
    const idx = base + ((color * NUM_PTYPE + pidx) * NUM_SQ + sq);
    x[idx] = 1.0;
  }

  // 持ち駒（枚数そのまま）
  const base2 = 1 + (2 * NUM_PTYPE * NUM_SQ);
  for (let i = 0; i < HAND_ORDER.length; i++) x[base2 + i] = (hands.S[HAND_ORDER[i]] || 0);
  for (let i = 0; i < HAND_ORDER.length; i++) x[base2 + HAND_ORDER.length + i] = (hands.G[HAND_ORDER[i]] || 0);

  return x;
}

/* NN推論（先手視点 -1..+1 を返す） */
function predictSenteValue(feats2283) {
  if (!tfModel || typeof tf === "undefined") return null;
  if (tfExpectedD && tfExpectedD !== D) return null;

  let v = null;
  tf.tidy(() => {
    const input = tf.tensor(feats2283, [1, D], "float32");

    let out;
    const dict = { [tfInputKey]: input };

    // GraphModel: execute（同期）でOKなことが多い
    out = tfOutputTensorName ? tfModel.execute(dict, tfOutputTensorName) : tfModel.execute(dict);

    const y = Array.isArray(out) ? out[0] : out;
    v = y.dataSync()[0];
  });
  return v;
}

/* ------------------ NN評価（negamax用：手番側から見たスコアを返す） ------------------ */
const evalCache = new Map(); // key -> score

function evalForSearch(board, hands, sideToMove, useNN) {
  const fallback = () => materialEval(board, hands) * (sideToMove === 'S' ? 1 : -1);

  if (!useNN || !tfModel || typeof tf === "undefined") return fallback();

  const key = positionKey(board, hands, sideToMove);
  const cached = evalCache.get(key);
  if (cached != null) return cached;

  const feats = encodePositionForNNUE(board, hands, sideToMove);
  const v = predictSenteValue(feats);
  if (v == null || Number.isNaN(v)) return fallback();

  const senteScore = v * 10000.0;
  const score = (sideToMove === 'S') ? senteScore : -senteScore;

  evalCache.set(key, score);
  return score;
}

/* ------------------ Game State ------------------ */
const PIECE_JA = {FU:"歩",KY:"香",KE:"桂",GI:"銀",KI:"金",KA:"角",HI:"飛",OU:"玉",TO:"と",NY:"杏",NK:"圭",NG:"全",UM:"馬",RY:"龍"};
const PROM_MAP = {FU:"TO", KY:"NY", KE:"NK", GI:"NG", KA:"UM", HI:"RY"};
const UNPROM_MAP = {TO:"FU", NY:"KY", NK:"KE", NG:"GI", UM:"KA", RY:"HI"};

let board, hands, sideToMove;
let selected = null;
let selectedDrop = null;
let history = [];

/* ------------------ UI ------------------ */
const elBoard = document.getElementById("board");
const elHandsS = document.getElementById("handsS");
const elHandsG = document.getElementById("handsG");
const elStatus = document.getElementById("status");
const elDepth = document.getElementById("depth");
const elUseNN = document.getElementById("useNN");

document.getElementById("btnReset").onclick = () => resetGame();
document.getElementById("btnUndo").onclick = () => undo();
document.getElementById("btnAIMove").onclick = () => aiMove();

function setStatus(extra = "") {
  const nn = elUseNN?.checked ? "ON" : "OFF";
  const legal = (board && hands && sideToMove) ? countLegalMoves(board, hands, sideToMove) : 0;
  const chk = (board && sideToMove && isKingInCheck(board, sideToMove)) ? "（王手）" : "";

  const inputInfo = tfInputKey ? `inputKey=${tfInputKey} (${tfInputTensorName || "?"})` : "inputKey=?";
  const outInfo = tfOutputTensorName ? `output=${tfOutputTensorName}` : "output=?";

  elStatus.textContent =
`手番: ${sideToMove || "-"} ${chk}
NN: ${nn} / モデル: ${tfModelStatus}
model: ${MODEL_URL}
${inputInfo}
${outInfo}
D(code): ${D} / D(model): ${tfExpectedD || "?"}
合法手数: ${legal}
${extra}`.trim();
}

function render() {
  elBoard.innerHTML = "";
  for (let y = 0; y < 9; y++) {
    const tr = document.createElement("tr");
    for (let x = 0; x < 9; x++) {
      const td = document.createElement("td");
      td.dataset.x = x; td.dataset.y = y;
      const p = board[y][x];
      if (p) {
        const s = PIECE_JA[p.type] || p.type;
        td.innerHTML = (p.side === 'G') ? `<span class="piece-g">${s}</span>` : s;
      } else td.textContent = "";

      if (selected && selected.x === x && selected.y === y) td.classList.add("sel");
      td.onclick = () => onSquareClick(x, y);
      tr.appendChild(td);
    }
    elBoard.appendChild(tr);
  }

  renderHands('S', elHandsS);
  renderHands('G', elHandsG);
  setStatus();
}

function renderHands(side, el) {
  el.innerHTML = "";
  const order = ['FU','KY','KE','GI','KI','KA','HI'];
  for (const t of order) {
    const n = hands[side][t] || 0;
    if (n <= 0) continue;
    const btn = document.createElement("button");
    btn.className = "handbtn" + (selectedDrop === t && sideToMove === side ? " sel" : "");
    btn.textContent = `${PIECE_JA[t]}×${n}`;
    btn.onclick = () => {
      if (sideToMove !== side) return;
      selected = null;
      selectedDrop = (selectedDrop === t ? null : t);
      render();
    };
    el.appendChild(btn);
  }
}

function onSquareClick(x, y) {
  if (selectedDrop) {
    if (board[y][x]) return;
    const mv = {from:null, to:{x,y}, drop:selectedDrop, promote:false};
    const legals = generateLegalMoves(board, hands, sideToMove);
    if (!legals.some(m => moveEq(m, mv))) return;
    pushHistory();
    ({board, hands, sideToMove} = applyMove(board, hands, sideToMove, mv));
    selectedDrop = null;
    selected = null;
    afterMove();
    return;
  }

  const p = board[y][x];

  if (!selected) {
    if (p && p.side === sideToMove) {
      selected = {x,y};
      render();
    }
    return;
  }

  if (p && p.side === sideToMove) {
    selected = {x,y};
    render();
    return;
  }

  const mvBase = {from:{...selected}, to:{x,y}, drop:null, promote:false};

  const legals = generateLegalMoves(board, hands, sideToMove);
  const cands = legals.filter(m => sameFromTo(m, mvBase) && !m.drop);
  if (cands.length === 0) return;

  let mv = cands[0];
  if (cands.length === 2) {
    const want = confirm("成りますか？（OK=成る / キャンセル=成らない）");
    mv = cands.find(m => m.promote === want) || mv;
  }

  pushHistory();
  ({board, hands, sideToMove} = applyMove(board, hands, sideToMove, mv));
  selected = null;
  afterMove();
}

function afterMove() {
  evalCache.clear();
  render();

  const legals = generateLegalMoves(board, hands, sideToMove);
  if (legals.length === 0) {
    const chk = isKingInCheck(board, sideToMove);
    alert(chk ? `詰み：${sideToMove}の負け` : "引き分け（合法手なし）");
  }
}

/* ------------------ Init ------------------ */
function resetGame() {
  board = makeEmptyBoard();
  hands = makeEmptyHands();
  sideToMove = 'S';
  selected = null;
  selectedDrop = null;
  history = [];
  setupInitialPosition(board);
  evalCache.clear();
  render();
}
function pushHistory() {
  history.push({
    board: cloneBoard(board),
    hands: cloneHands(hands),
    side: sideToMove
  });
}
function undo() {
  const st = history.pop();
  if (!st) return;
  board = st.board; hands = st.hands; sideToMove = st.side;
  selected = null; selectedDrop = null;
  evalCache.clear();
  render();
}
resetGame();

/* ------------------ Board / Hands helpers ------------------ */
function makeEmptyBoard(){ return Array.from({length:9},()=>Array(9).fill(null)); }
function cloneBoard(b){ return b.map(row=>row.map(p=>p?{...p}:null)); }
function makeEmptyHands(){
  return { S:{FU:0,KY:0,KE:0,GI:0,KI:0,KA:0,HI:0}, G:{FU:0,KY:0,KE:0,GI:0,KI:0,KA:0,HI:0} };
}
function cloneHands(h){
  const out = makeEmptyHands();
  for (const s of ['S','G']) for (const k in out[s]) out[s][k] = h[s][k]||0;
  return out;
}
function setupInitialPosition(b){
  const G = 'G', S = 'S';
  b[0][0]={side:G,type:'KY'}; b[0][1]={side:G,type:'KE'}; b[0][2]={side:G,type:'GI'}; b[0][3]={side:G,type:'KI'};
  b[0][4]={side:G,type:'OU'}; b[0][5]={side:G,type:'KI'}; b[0][6]={side:G,type:'GI'}; b[0][7]={side:G,type:'KE'}; b[0][8]={side:G,type:'KY'};
  b[1][1]={side:G,type:'HI'}; b[1][7]={side:G,type:'KA'};
  for (let x=0;x<9;x++) b[2][x]={side:G,type:'FU'};

  b[8][0]={side:S,type:'KY'}; b[8][1]={side:S,type:'KE'}; b[8][2]={side:S,type:'GI'}; b[8][3]={side:S,type:'KI'};
  b[8][4]={side:S,type:'OU'}; b[8][5]={side:S,type:'KI'}; b[8][6]={side:S,type:'GI'}; b[8][7]={side:S,type:'KE'}; b[8][8]={side:S,type:'KY'};
  b[7][1]={side:S,type:'KA'}; b[7][7]={side:S,type:'HI'};
  for (let x=0;x<9;x++) b[6][x]={side:S,type:'FU'};
}

/* ------------------ Moves ------------------ */
function sameFromTo(a, b){
  return a.from && b.from && a.from.x===b.from.x && a.from.y===b.from.y && a.to.x===b.to.x && a.to.y===b.to.y;
}
function moveEq(a, b){
  if (!!a.drop !== !!b.drop) return false;
  if (a.drop) return a.drop===b.drop && a.to.x===b.to.x && a.to.y===b.to.y;
  return sameFromTo(a,b) && !!a.promote===!!b.promote;
}
function inside(x,y){ return x>=0&&x<9&&y>=0&&y<9; }
function opponent(side){ return side==='S'?'G':'S'; }

function applyMove(b, h, side, mv){
  const nb = cloneBoard(b);
  const nh = cloneHands(h);

  if (mv.drop) {
    nb[mv.to.y][mv.to.x] = {side, type: mv.drop};
    nh[side][mv.drop]--;
    return {board: nb, hands: nh, sideToMove: opponent(side)};
  }

  const fromP = nb[mv.from.y][mv.from.x];
  const toP = nb[mv.to.y][mv.to.x];

  if (toP) {
    const base = UNPROM_MAP[toP.type] || toP.type;
    if (base !== 'OU') nh[side][base] = (nh[side][base]||0) + 1;
  }

  nb[mv.from.y][mv.from.x] = null;

  let newType = fromP.type;
  if (mv.promote) newType = PROM_MAP[fromP.type] || fromP.type;

  nb[mv.to.y][mv.to.x] = {side, type: newType};

  return {board: nb, hands: nh, sideToMove: opponent(side)};
}

/* ------------------ Legal move generation ------------------ */
function generateLegalMoves(b, h, side){
  const moves = [];

  for (let y=0;y<9;y++) for (let x=0;x<9;x++) {
    const p = b[y][x];
    if (!p || p.side !== side) continue;
    const pm = generatePseudoMovesForPiece(b, x, y, p);
    for (const m of pm) moves.push(m);
  }

  const dropMoves = generateDropMoves(b, h, side);
  for (const m of dropMoves) moves.push(m);

  const legals = [];
  for (const mv of moves) {
    const st = applyMove(b, h, side, mv);
    if (!isKingInCheck(st.board, side)) {
      if (mv.drop === 'FU') {
        if (isKingInCheck(st.board, opponent(side))) {
          const reply = generateLegalMoves(st.board, st.hands, opponent(side));
          if (reply.length === 0) continue;
        }
      }
      legals.push(mv);
    }
  }
  return legals;
}
function countLegalMoves(b,h,side){ return generateLegalMoves(b,h,side).length; }

function inPromoZone(side, y){
  return side==='S' ? (y<=2) : (y>=6);
}
function mustPromote(pieceType, side, toY){
  if (pieceType === 'FU' || pieceType === 'KY') {
    return side==='S' ? (toY===0) : (toY===8);
  }
  if (pieceType === 'KE') {
    return side==='S' ? (toY<=1) : (toY>=7);
  }
  return false;
}
function canPromote(pieceType){
  return !!PROM_MAP[pieceType];
}

function generatePseudoMovesForPiece(b, x, y, p){
  const res = [];
  const side = p.side;
  const dir = (side === 'S') ? -1 : 1;

  const addStep = (dx, dy) => {
    const nx = x + dx, ny = y + dy;
    if (!inside(nx,ny)) return;
    const tp = b[ny][nx];
    if (tp && tp.side === side) return;
    pushMoveWithPromo(res, {from:{x,y}, to:{x:nx,y:ny}, drop:null, promote:false}, p.type, side, y, ny);
  };

  const addSlide = (dx, dy) => {
    for (let k=1;k<9;k++){
      const nx = x + dx*k, ny = y + dy*k;
      if (!inside(nx,ny)) break;
      const tp = b[ny][nx];
      if (tp && tp.side === side) break;
      pushMoveWithPromo(res, {from:{x,y}, to:{x:nx,y:ny}, drop:null, promote:false}, p.type, side, y, ny);
      if (tp) break;
    }
  };

  switch(p.type){
    case 'FU': addStep(0, dir); break;
    case 'KY': addSlide(0, dir); break;
    case 'KE': addStep(-1, 2*dir); addStep(1, 2*dir); break;
    case 'GI':
      addStep(-1, dir); addStep(0, dir); addStep(1, dir);
      addStep(-1, -dir); addStep(1, -dir);
      break;
    case 'KI':
    case 'TO': case 'NY': case 'NK': case 'NG':
      addStep(-1, dir); addStep(0, dir); addStep(1, dir);
      addStep(-1, 0); addStep(1, 0);
      addStep(0, -dir);
      break;
    case 'KA': addSlide(1,1); addSlide(1,-1); addSlide(-1,1); addSlide(-1,-1); break;
    case 'HI': addSlide(1,0); addSlide(-1,0); addSlide(0,1); addSlide(0,-1); break;
    case 'OU':
      for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) if (dx||dy) addStep(dx,dy);
      break;
    case 'UM':
      addSlide(1,1); addSlide(1,-1); addSlide(-1,1); addSlide(-1,-1);
      addStep(1,0); addStep(-1,0); addStep(0,1); addStep(0,-1);
      break;
    case 'RY':
      addSlide(1,0); addSlide(-1,0); addSlide(0,1); addSlide(0,-1);
      addStep(1,1); addStep(1,-1); addStep(-1,1); addStep(-1,-1);
      break;
  }
  return res;
}

function pushMoveWithPromo(arr, mv, pieceType, side, fromY, toY){
  if (!canPromote(pieceType)) { arr.push(mv); return; }

  const promoPossible = inPromoZone(side, fromY) || inPromoZone(side, toY);
  const forced = mustPromote(pieceType, side, toY);

  if (!promoPossible) { arr.push(mv); return; }
  if (forced) { arr.push({...mv, promote:true}); return; }

  arr.push({...mv, promote:false});
  arr.push({...mv, promote:true});
}

function generateDropMoves(b, h, side){
  const res = [];
  const order = ['FU','KY','KE','GI','KI','KA','HI'];

  for (const t of order) {
    if ((h[side][t]||0) <= 0) continue;

    for (let y=0;y<9;y++) for (let x=0;x<9;x++) {
      if (b[y][x]) continue;

      if (t === 'FU' || t === 'KY') {
        if ((side==='S' && y===0) || (side==='G' && y===8)) continue;
      }
      if (t === 'KE') {
        if ((side==='S' && y<=1) || (side==='G' && y>=7)) continue;
      }

      if (t === 'FU') {
        if (hasUnpromotedPawnOnFile(b, side, x)) continue;
      }

      res.push({from:null, to:{x,y}, drop:t, promote:false});
    }
  }
  return res;
}
function hasUnpromotedPawnOnFile(b, side, fileX){
  for (let y=0;y<9;y++){
    const p = b[y][fileX];
    if (p && p.side===side && p.type==='FU') return true;
  }
  return false;
}

/* ------------------ Check detection ------------------ */
function findKing(b, side){
  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=b[y][x];
    if (p && p.side===side && p.type==='OU') return {x,y};
  }
  return null;
}
function isKingInCheck(b, side){
  const k = findKing(b, side);
  if (!k) return false;
  const foe = opponent(side);

  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=b[y][x];
    if (!p || p.side!==foe) continue;
    if (attacksSquare(b, x, y, p, k.x, k.y)) return true;
  }
  return false;
}
function attacksSquare(b, x, y, p, tx, ty){
  const side = p.side;
  const dir = (side === 'S') ? -1 : 1;

  const step = (dx,dy) => (x+dx===tx && y+dy===ty);
  const slide = (dx,dy) => {
    for (let k=1;k<9;k++){
      const nx=x+dx*k, ny=y+dy*k;
      if (!inside(nx,ny)) return false;
      if (nx===tx && ny===ty) return true;
      if (b[ny][nx]) return false;
    }
    return false;
  };

  switch(p.type){
    case 'FU': return step(0,dir);
    case 'KY': return slide(0,dir);
    case 'KE': return step(-1,2*dir) || step(1,2*dir);
    case 'GI':
      return step(-1,dir)||step(0,dir)||step(1,dir)||step(-1,-dir)||step(1,-dir);
    case 'KI':
    case 'TO': case 'NY': case 'NK': case 'NG':
      return step(-1,dir)||step(0,dir)||step(1,dir)||step(-1,0)||step(1,0)||step(0,-dir);
    case 'KA': return slide(1,1)||slide(1,-1)||slide(-1,1)||slide(-1,-1);
    case 'HI': return slide(1,0)||slide(-1,0)||slide(0,1)||slide(0,-1);
    case 'OU':
      for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) if (dx||dy) if (step(dx,dy)) return true;
      return false;
    case 'UM':
      return (slide(1,1)||slide(1,-1)||slide(-1,1)||slide(-1,-1)) ||
             step(1,0)||step(-1,0)||step(0,1)||step(0,-1);
    case 'RY':
      return (slide(1,0)||slide(-1,0)||slide(0,1)||slide(0,-1)) ||
             step(1,1)||step(1,-1)||step(-1,1)||step(-1,-1);
  }
  return false;
}

/* ------------------ Evaluation fallback ------------------ */
const VALUE = {FU:100,KY:300,KE:300,GI:400,KI:500,KA:700,HI:800,OU:0,TO:500,NY:500,NK:500,NG:500,UM:900,RY:1000};
function materialEval(b,h){
  let s=0;
  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=b[y][x]; if (!p) continue;
    const v = VALUE[p.type]||0;
    s += (p.side==='S') ? v : -v;
  }
  for (const side of ['S','G']){
    for (const t of ['FU','KY','KE','GI','KI','KA','HI']){
      const v = VALUE[t]||0;
      const n = h[side][t]||0;
      s += (side==='S') ? v*n : -v*n;
    }
  }
  return s;
}

/* ------------------ Search (negamax + alpha-beta) ------------------ */
function positionKey(b,h,side){
  let s = side + "|";
  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=b[y][x];
    if (!p) s += ".";
    else s += (p.side==='S'?"S":"G")+p.type;
    s += ",";
  }
  s += "|";
  for (const sd of ['S','G']){
    s += sd + ":";
    for (const t of ['FU','KY','KE','GI','KI','KA','HI']) s += (h[sd][t]||0)+".";
    s += "|";
  }
  return s;
}

// 簡易の手順序
function orderMoves(b,h,side,moves){
  const foe = opponent(side);
  const scored = moves.map(mv => {
    let sc = 0;
    if (!mv.drop) {
      const tp = b[mv.to.y][mv.to.x];
      if (tp) sc += (VALUE[tp.type] || 0) + 200;
      if (mv.promote) sc += 150;
    } else {
      sc += 20;
    }
    const st = applyMove(b,h,side,mv);
    if (isKingInCheck(st.board, foe)) sc += 400;
    return {mv, sc};
  });
  scored.sort((a,b)=>b.sc-a.sc);
  return scored.map(x=>x.mv);
}

function chooseBestMove(b,h,side,depth,useNN){
  let moves = generateLegalMoves(b,h,side);
  if (moves.length===0) return null;
  moves = orderMoves(b,h,side,moves);

  let best = moves[0];
  let bestScore = -Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (const mv of moves){
    const st = applyMove(b,h,side,mv);
    const score = -negamax(st.board, st.hands, st.sideToMove, depth-1, -beta, -alpha, useNN);
    if (score > bestScore){
      bestScore = score; best = mv;
    }
    if (score > alpha) alpha = score;
  }
  return {move: best, score: bestScore};
}

function negamax(b,h,side,depth,alpha,beta,useNN){
  let legals = generateLegalMoves(b,h,side);
  if (legals.length===0){
    if (isKingInCheck(b,side)) return -999999;
    return 0;
  }
  if (depth<=0){
    return evalForSearch(b,h,side,useNN);
  }

  legals = orderMoves(b,h,side,legals);

  let best = -Infinity;
  for (const mv of legals){
    const st = applyMove(b,h,side,mv);
    const score = -negamax(st.board, st.hands, st.sideToMove, depth-1, -beta, -alpha, useNN);
    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }
  return best;
}

/* ------------------ AI Move ------------------ */
function aiMove(){
  const depth = Math.max(1, Math.min(7, parseInt(elDepth.value||"3",10)));
  const useNN = !!elUseNN.checked;

  setStatus("AI思考中...");
  setTimeout(() => {
    const r = chooseBestMove(board, hands, sideToMove, depth, useNN);
    if (!r) return;
    pushHistory();
    ({board, hands, sideToMove} = applyMove(board, hands, sideToMove, r.move));
    selected = null; selectedDrop = null;
    afterMove();
  }, 30);
}
