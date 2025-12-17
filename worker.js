/* worker.js v4
   - tfjs を worker で読み込み
   - model.json を worker で読み込み
   - 反復深化 + αβ探索 + 時間打ち切り
*/

let tf = null;
let model = null;
let modelStatus = "未読込";

const VALUE = {FU:100,KY:300,KE:300,GI:400,KI:500,KA:700,HI:800,OU:0,TO:500,NY:500,NK:500,NG:500,UM:900,RY:1000};
const PROM_MAP = {FU:"TO", KY:"NY", KE:"NK", GI:"NG", KA:"UM", HI:"RY"};
const UNPROM_MAP = {TO:"FU", NY:"KY", NK:"KE", NG:"GI", UM:"KA", RY:"HI"};

const NUM_PTYPE = 14, NUM_SQ = 81;
const HAND_ORDER = ['FU','KY','KE','GI','KI','KA','HI'];
const D = 1 + (2 * NUM_PTYPE * NUM_SQ) + (2 * HAND_ORDER.length);

const PTYPE_ID = {
  FU: 1, KY: 2, KE: 3, GI: 4, KI: 5, KA: 6, HI: 7,
  OU: 8, TO: 9, NY: 10, NK: 11, NG: 12, UM: 13, RY: 14,
};

// あなたの答え合わせ済み：python側と一致
function sqIndex(x, y) { return y*9 + x; }

let inputKey = "input_layer_5";
let outputName = "Identity:0";
let expectedD = 2283;

let cancelRequestId = -1;

// ----- util -----
function cloneBoard(b){ return b.map(row=>row.map(p=>p?{...p}:null)); }
function cloneHands(h){
  const out = { S:{FU:0,KY:0,KE:0,GI:0,KI:0,KA:0,HI:0}, G:{FU:0,KY:0,KE:0,GI:0,KI:0,KA:0,HI:0} };
  for (const s of ['S','G']) for (const k in out[s]) out[s][k] = h[s][k]||0;
  return out;
}
function inside(x,y){ return x>=0&&x<9&&y>=0&&y<9; }
function opponent(side){ return side==='S'?'G':'S'; }

// ----- encode (2283) -----
function encodePositionForNNUE(board, hands, sideToMove) {
  const x = new Float32Array(D);
  x[0] = (sideToMove === 'S') ? 1.0 : 0.0;

  const base = 1;
  for (let y=0;y<9;y++) for (let x0=0;x0<9;x0++){
    const p = board[y][x0];
    if (!p) continue;
    const color = (p.side === 'S') ? 0 : 1;
    const pt = PTYPE_ID[p.type] || 0;
    if (!pt) continue;
    const sq = sqIndex(x0,y);
    const pidx = pt - 1;
    const idx = base + ((color * NUM_PTYPE + pidx) * NUM_SQ + sq);
    x[idx] = 1.0;
  }

  const base2 = 1 + (2 * NUM_PTYPE * NUM_SQ);
  for (let i=0;i<HAND_ORDER.length;i++) x[base2+i] = (hands.S[HAND_ORDER[i]]||0);
  for (let i=0;i<HAND_ORDER.length;i++) x[base2+HAND_ORDER.length+i] = (hands.G[HAND_ORDER[i]]||0);
  return x;
}

function predictSenteValue(feats2283) {
  if (!model || !tf) return null;
  let v = null;
  tf.tidy(() => {
    const input = tf.tensor(feats2283, [1, D], "float32");
    const dict = { [inputKey]: input };
    const out = outputName ? model.execute(dict, outputName) : model.execute(dict);
    const y = Array.isArray(out) ? out[0] : out;
    v = y.dataSync()[0];
  });
  return v;
}

// ----- eval -----
function materialEval(board, hands){
  let s=0;
  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=board[y][x]; if(!p) continue;
    const v = VALUE[p.type]||0;
    s += (p.side==='S') ? v : -v;
  }
  for (const side of ['S','G']){
    for (const t of HAND_ORDER){
      const v = VALUE[t]||0;
      const n = hands[side][t]||0;
      s += (side==='S') ? v*n : -v*n;
    }
  }
  return s;
}

function evalForSearch(board, hands, sideToMove, useNN){
  if (!useNN || !model || !tf) {
    const m = materialEval(board, hands);
    return (sideToMove==='S') ? m : -m;
  }
  const feats = encodePositionForNNUE(board, hands, sideToMove);
  const v = predictSenteValue(feats);
  if (v == null || Number.isNaN(v)) {
    const m = materialEval(board, hands);
    return (sideToMove==='S') ? m : -m;
  }
  const senteScore = v * 10000.0;
  return (sideToMove==='S') ? senteScore : -senteScore;
}

// ----- moves -----
function sameFromTo(a, b){
  return a.from && b.from && a.from.x===b.from.x && a.from.y===b.from.y && a.to.x===b.to.x && a.to.y===b.to.y;
}
function moveEq(a, b){
  if (!!a.drop !== !!b.drop) return false;
  if (a.drop) return a.drop===b.drop && a.to.x===b.to.x && a.to.y===b.to.y;
  return sameFromTo(a,b) && !!a.promote===!!b.promote;
}

function applyMove(board, hands, side, mv){
  const nb = cloneBoard(board);
  const nh = cloneHands(hands);

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

function inPromoZone(side, y){ return side==='S' ? (y<=2) : (y>=6); }
function mustPromote(pieceType, side, toY){
  if (pieceType === 'FU' || pieceType === 'KY') return side==='S' ? (toY===0) : (toY===8);
  if (pieceType === 'KE') return side==='S' ? (toY<=1) : (toY>=7);
  return false;
}
function canPromote(pieceType){ return !!PROM_MAP[pieceType]; }

function pushMoveWithPromo(arr, mv, pieceType, side, fromY, toY){
  if (!canPromote(pieceType)) { arr.push(mv); return; }
  const promoPossible = inPromoZone(side, fromY) || inPromoZone(side, toY);
  const forced = mustPromote(pieceType, side, toY);
  if (!promoPossible) { arr.push(mv); return; }
  if (forced) { arr.push({...mv, promote:true}); return; }
  arr.push({...mv, promote:false});
  arr.push({...mv, promote:true});
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

function hasUnpromotedPawnOnFile(b, side, fileX){
  for (let y=0;y<9;y++){
    const p = b[y][fileX];
    if (p && p.side===side && p.type==='FU') return true;
  }
  return false;
}

function generateDropMoves(b, h, side){
  const res = [];
  const order = HAND_ORDER;
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

function findKing(b, side){
  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p=b[y][x];
    if (p && p.side===side && p.type==='OU') return {x,y};
  }
  return null;
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

function generateLegalMoves(b, h, side){
  const moves = [];
  for (let y=0;y<9;y++) for (let x=0;x<9;x++) {
    const p = b[y][x];
    if (!p || p.side !== side) continue;
    const pm = generatePseudoMovesForPiece(b, x, y, p);
    for (const m of pm) moves.push(m);
  }
  for (const m of generateDropMoves(b, h, side)) moves.push(m);

  const legals = [];
  for (const mv of moves) {
    const st = applyMove(b, h, side, mv);
    if (!isKingInCheck(st.board, side)) {
      if (mv.drop === 'FU') {
        if (isKingInCheck(st.board, opponent(side))) {
          const reply = generateLegalMoves(st.board, st.hands, opponent(side));
          if (reply.length === 0) continue; // 打ち歩詰め簡易
        }
      }
      legals.push(mv);
    }
  }
  return legals;
}

// 手順序（簡易）
function orderMoves(b,h,side,moves, pvMove){
  const foe = opponent(side);
  const scored = moves.map(mv => {
    let sc = 0;

    if (pvMove && moveEq(mv, pvMove)) sc += 100000; // PV最優先

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

// ----- search -----
function timeOrCancel(deadline, requestId){
  if (requestId === cancelRequestId) throw new Error("__CANCEL__");
  if (Date.now() >= deadline) throw new Error("__TIME__");
}

function negamax(b,h,side,depth,alpha,beta,useNN,deadline,requestId, pvMove){
  timeOrCancel(deadline, requestId);

  const legals0 = generateLegalMoves(b,h,side);
  if (legals0.length===0){
    if (isKingInCheck(b,side)) return -999999;
    return 0;
  }
  if (depth<=0) return evalForSearch(b,h,side,useNN);

  let legals = orderMoves(b,h,side,legals0, pvMove);

  let best = -Infinity;
  let bestLocalMove = null;

  for (const mv of legals){
    const st = applyMove(b,h,side,mv);
    const score = -negamax(st.board, st.hands, st.sideToMove, depth-1, -beta, -alpha, useNN, deadline, requestId, null);
    if (score > best){
      best = score;
      bestLocalMove = mv;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }

  // ルート以外ではmoveは不要（返さない）
  return best;
}

function chooseBestMoveRoot(b,h,side,depth,useNN,deadline,requestId, pvMove){
  timeOrCancel(deadline, requestId);

  const legals0 = generateLegalMoves(b,h,side);
  if (legals0.length===0) return null;

  const legals = orderMoves(b,h,side,legals0, pvMove);

  let bestMove = legals[0];
  let bestScore = -Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (const mv of legals){
    const st = applyMove(b,h,side,mv);
    const score = -negamax(st.board, st.hands, st.sideToMove, depth-1, -beta, -alpha, useNN, deadline, requestId, null);
    if (score > bestScore){
      bestScore = score;
      bestMove = mv;
    }
    if (score > alpha) alpha = score;
  }

  return {move: bestMove, score: bestScore};
}

function toUci(mv){
  if (!mv) return "";
  if (mv.drop) return `D*${mv.drop}@${mv.to.x}${mv.to.y}`; // デバッグ用
  return `${mv.from.x}${mv.from.y}${mv.to.x}${mv.to.y}${mv.promote ? "+" : ""}`;
}

// ----- init model -----
async function initTF(modelUrl, backend){
  try {
    modelStatus = "tfjs読込中...";

    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");
    tf = self.tf;

    if (backend === "wasm") {
      importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js");
      tf.wasm.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
      await tf.setBackend("wasm");
      await tf.ready();
    } else {
      await tf.setBackend("cpu");
      await tf.ready();
    }

    modelStatus = "model読込中...";
    model = await tf.loadGraphModel(modelUrl, { fromTFHub: false });

    // signatureを読めたら更新（読めなくても固定値で動く）
    try {
      const meta = await (await fetch(modelUrl, { cache: "no-store" })).json();
      const ins = meta?.signature?.inputs || null;
      const outs = meta?.signature?.outputs || null;
      const inKeys = ins ? Object.keys(ins) : [];
      const outKeys = outs ? Object.keys(outs) : [];
      if (inKeys.length) {
        inputKey = inKeys[0];
        expectedD = parseInt(ins[inputKey]?.tensorShape?.dim?.[1]?.size || expectedD, 10);
      }
      if (outKeys.length) outputName = outs[outKeys[0]]?.name || outputName;
    } catch (e) {
      // ignore
    }

    // warmup
    tf.tidy(() => {
      const z = tf.zeros([1, D], "float32");
      const dict = { [inputKey]: z };
      const out = outputName ? model.execute(dict, outputName) : model.execute(dict);
      const y = Array.isArray(out) ? out[0] : out;
      y.dataSync();
    });

    modelStatus = "読込完了";
    return { ok:true, info:`backend=${tf.getBackend()} inputKey=${inputKey} D=${expectedD}` };
  } catch (e) {
    model = null;
    modelStatus = "読込失敗";
    return { ok:false, error: (e?.message || String(e)) };
  }
}

// ----- message loop -----
self.onmessage = async (ev) => {
  const msg = ev.data || {};

  if (msg.type === "init") {
    const r = await initTF(msg.modelUrl, msg.backend || "wasm");
    self.postMessage({ type:"init_done", ok:r.ok, error:r.error, info:r.info || "" });
    return;
  }

  if (msg.type === "cancel") {
    cancelRequestId = msg.requestId;
    return;
  }

  if (msg.type === "think") {
    const requestId = msg.requestId;
    const depthMax = msg.depthMax || 4;
    const timeMs = msg.timeMs || 500;
    const useNN = !!msg.useNN;

    const pos = msg.pos;
    const b0 = pos.board;
    const h0 = pos.hands;
    const side0 = pos.sideToMove;

    const start = Date.now();
    const deadline = start + timeMs;

    let bestMove = null;
    let bestScore = null;
    let depthDone = 0;
    let pvMove = null;

    try {
      for (let d=1; d<=depthMax; d++){
        timeOrCancel(deadline, requestId);

        const r = chooseBestMoveRoot(b0, h0, side0, d, useNN, deadline, requestId, pvMove);
        if (r && r.move) {
          bestMove = r.move;
          bestScore = r.score;
          pvMove = r.move;
          depthDone = d;

          self.postMessage({
            type:"progress",
            requestId,
            depthDone,
            depthMax,
            bestScore,
            bestUci: toUci(bestMove),
          });
        } else {
          break;
        }
      }
    } catch (e) {
      // __TIME__ / __CANCEL__ は想定内
    }

    self.postMessage({
      type:"result",
      requestId,
      ok: !!bestMove,
      bestMove,
      bestScore,
      depthDone,
      timeMs: Date.now() - start,
      modelStatus
    });
    return;
  }
};
