/* worker.js (vTT1)
   - TFJS model load in Worker
   - Iterative deepening + Aspiration window
   - NN eval cache
   - Move ordering 강화: TT move first, captures (MVV-LVA), promotions, checks, killer, history
   - Transposition Table (TT): depth-bound, exact/lower/upper

   Protocol:
     main -> worker:
       {type:"init", modelUrl, backend:"wasm"|"cpu"}
       {type:"think", requestId, depthMax, timeMs, useNN, pos:{board,hands,sideToMove}}

     worker -> main:
       {type:"init_done", ok, info, error}
       {type:"progress", requestId, depthDone, depthMax, bestMove, bestStr, bestScore, partial}
       {type:"result", requestId, ok, depthDone, depthMax, bestMove, bestStr, bestScore, timeMs, partial, nodes}
*/

"use strict";

/* ------------------ TFJS Load ------------------ */
let tfReady = false;
let model = null;
let modelInfo = "";
let backendName = "cpu";

async function initTf(modelUrl, backend) {
  try {
    // TFJS core
    importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");

    // backend
    if (backend === "wasm") {
      importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js");
      // wasm binaries path
      if (tf?.wasm?.setWasmPaths) {
        tf.wasm.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
      }
      await tf.setBackend("wasm");
      await tf.ready();
      backendName = "wasm";
    } else {
      await tf.setBackend("cpu");
      await tf.ready();
      backendName = "cpu";
    }

    model = await tf.loadLayersModel(modelUrl);

    const inName = model?.inputs?.[0]?.name || "(unknown)";
    const outName = model?.outputs?.[0]?.name || "(unknown)";
    const inShape = model?.inputs?.[0]?.shape ? JSON.stringify(model.inputs[0].shape) : "(unknown)";
    modelInfo =
      `model: ${modelUrl}\n` +
      `backend: ${backendName}\n` +
      `input: ${inName} shape=${inShape}\n` +
      `output: ${outName}\n`;

    tfReady = true;
    return { ok: true };
  } catch (e) {
    tfReady = false;
    model = null;
    return { ok: false, error: String(e?.message || e) };
  }
}

/* ------------------ Shogi internal representation ------------------ */
/*
  boardA[sq] : Int16
    0 = empty
    +ptype (1..14) = S piece
    -ptype (1..14) = G piece

  sideSign : +1 (S to move) / -1 (G to move)

  piece types (match python-shogi piece_type 1..14 order):
    1 FU, 2 KY, 3 KE, 4 GI, 5 KI, 6 KA, 7 HI, 8 OU,
    9 TO, 10 NY, 11 NK, 12 NG, 13 UM, 14 RY
*/
const PTYPE = {
  FU: 1, KY: 2, KE: 3, GI: 4, KI: 5, KA: 6, HI: 7, OU: 8,
  TO: 9, NY: 10, NK: 11, NG: 12, UM: 13, RY: 14,
};
const PNAME = {
  1:"FU",2:"KY",3:"KE",4:"GI",5:"KI",6:"KA",7:"HI",8:"OU",
  9:"TO",10:"NY",11:"NK",12:"NG",13:"UM",14:"RY"
};

const HAND_ORDER = [PTYPE.FU, PTYPE.KY, PTYPE.KE, PTYPE.GI, PTYPE.KI, PTYPE.KA, PTYPE.HI]; // 7
const HAND_IDX = new Map([[1,0],[2,1],[3,2],[4,3],[5,4],[6,5],[7,6]]);

const PROM = new Int8Array(15);
PROM[PTYPE.FU] = PTYPE.TO;
PROM[PTYPE.KY] = PTYPE.NY;
PROM[PTYPE.KE] = PTYPE.NK;
PROM[PTYPE.GI] = PTYPE.NG;
PROM[PTYPE.KA] = PTYPE.UM;
PROM[PTYPE.HI] = PTYPE.RY;

const UNPROM = new Int8Array(15);
UNPROM[PTYPE.TO] = PTYPE.FU;
UNPROM[PTYPE.NY] = PTYPE.KY;
UNPROM[PTYPE.NK] = PTYPE.KE;
UNPROM[PTYPE.NG] = PTYPE.GI;
UNPROM[PTYPE.UM] = PTYPE.KA;
UNPROM[PTYPE.RY] = PTYPE.HI;

function baseTypeOf(pt) {
  return UNPROM[pt] ? UNPROM[pt] : pt;
}
function sideIdxFromSign(sign){ return sign === 1 ? 0 : 1; }
function oppSign(sign){ return -sign; }
function sqX(sq){ return sq % 9; }
function sqY(sq){ return (sq / 9) | 0; }
function sqIndex(x,y){ return y*9 + x; }
function inside(x,y){ return x>=0 && x<9 && y>=0 && y<9; }

function inPromoZone(sign, y){
  // S: y<=2, G: y>=6
  return (sign === 1) ? (y <= 2) : (y >= 6);
}
function mustPromote(pt, sign, toY){
  if (pt === PTYPE.FU || pt === PTYPE.KY) {
    return (sign===1) ? (toY===0) : (toY===8);
  }
  if (pt === PTYPE.KE) {
    return (sign===1) ? (toY<=1) : (toY>=7);
  }
  return false;
}
function canPromote(pt){
  return PROM[pt] !== 0;
}

/* ------------------ Values (ordering / fallback) ------------------ */
const VALUE = new Int32Array(15);
VALUE[PTYPE.FU]=100; VALUE[PTYPE.KY]=300; VALUE[PTYPE.KE]=300; VALUE[PTYPE.GI]=400; VALUE[PTYPE.KI]=500;
VALUE[PTYPE.KA]=700; VALUE[PTYPE.HI]=800; VALUE[PTYPE.OU]=0;
VALUE[PTYPE.TO]=500; VALUE[PTYPE.NY]=500; VALUE[PTYPE.NK]=500; VALUE[PTYPE.NG]=500;
VALUE[PTYPE.UM]=900; VALUE[PTYPE.RY]=1000;

/* ------------------ Zobrist (for TT / eval cache) ------------------ */
let Z_PIECE = null; // [2][15][81] BigInt
let Z_HAND  = null; // [2][7][20] BigInt (count 0..19)
let Z_SIDE  = 0n;

function splitmix64(seed) {
  // returns [nextSeed, rand64BigInt]
  let x = BigInt.asUintN(64, seed + 0x9E3779B97F4A7C15n);
  seed = x;
  x = BigInt.asUintN(64, (x ^ (x >> 30n)) * 0xBF58476D1CE4E5B9n);
  x = BigInt.asUintN(64, (x ^ (x >> 27n)) * 0x94D049BB133111EBn);
  x = BigInt.asUintN(64, x ^ (x >> 31n));
  return [seed, x];
}
function initZobrist() {
  let seed = 0x123456789ABCDEFn;

  Z_PIECE = Array.from({length:2}, () =>
    Array.from({length:15}, () => Array.from({length:81}, () => 0n))
  );
  Z_HAND = Array.from({length:2}, () =>
    Array.from({length:7}, () => Array.from({length:20}, () => 0n))
  );

  for (let s=0;s<2;s++){
    for (let pt=1;pt<=14;pt++){
      for (let sq=0;sq<81;sq++){
        [seed, Z_PIECE[s][pt][sq]] = splitmix64(seed);
      }
    }
  }
  for (let s=0;s<2;s++){
    for (let i=0;i<7;i++){
      for (let c=0;c<20;c++){
        [seed, Z_HAND[s][i][c]] = splitmix64(seed);
      }
    }
  }
  [seed, Z_SIDE] = splitmix64(seed);
}
initZobrist();

/* ------------------ State (in worker) ------------------ */
let boardA = new Int16Array(81);
let handsA = [new Int16Array(7), new Int16Array(7)]; // [S][7], [G][7]
let sideSign = 1; // +1 S, -1 G
let hashKey = 0n;

// undo stack
const undoStack = [];

/* build hash from scratch */
function computeHash() {
  let h = 0n;
  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    const s = p > 0 ? 0 : 1;
    const pt = Math.abs(p);
    h ^= Z_PIECE[s][pt][sq];
  }
  for (let s=0;s<2;s++){
    for (let i=0;i<7;i++){
      const c = handsA[s][i];
      const cc = Math.max(0, Math.min(19, c));
      h ^= Z_HAND[s][i][cc];
    }
  }
  if (sideSign === -1) h ^= Z_SIDE; // side bit
  return h;
}

/* ------------------ Move encoding ------------------ */
/*
  move object:
    {fromSq, toSq, dropPt(0|1..7), promote(0|1), movedPt, capturedPiece, givesCheck, key}
*/
function moveKey(mv){
  if (mv.dropPt) {
    // 0x80000000 | (dropPt<<16) | to
    return (0x80000000 | (mv.dropPt << 16) | (mv.toSq & 0xFF)) >>> 0;
  }
  // (from<<16) | to | (promote<<30)
  return (((mv.fromSq & 0xFF) << 16) | (mv.toSq & 0xFF) | ((mv.promote?1:0) << 30)) >>> 0;
}
function sameMoveKey(aKey, bKey){ return aKey === bKey; }

function mvToBestStr(mv){
  const pad2 = (n)=> String(n).padStart(2,"0");
  if (mv.dropPt){
    return `D${PNAME[mv.dropPt] || mv.dropPt}${pad2(mv.toSq)}`;
  }
  return `${pad2(mv.fromSq)}${pad2(mv.toSq)}${mv.promote?"+":""}`;
}

function mvToUI(mv){
  const to = { x: sqX(mv.toSq), y: sqY(mv.toSq) };
  if (mv.dropPt){
    return { from: null, to, drop: (PNAME[mv.dropPt] || "FU"), promote: false };
  }
  const from = { x: sqX(mv.fromSq), y: sqY(mv.fromSq) };
  return { from, to, drop: null, promote: !!mv.promote };
}

/* ------------------ Make / Unmake (in-place) ------------------ */
function xorHandCount(sign, handIdx, oldC, newC){
  const s = sideIdxFromSign(sign);
  const o = Math.max(0, Math.min(19, oldC));
  const n = Math.max(0, Math.min(19, newC));
  hashKey ^= Z_HAND[s][handIdx][o];
  hashKey ^= Z_HAND[s][handIdx][n];
}

function makeMove(mv){
  // store undo
  const u = {
    fromSq: mv.fromSq,
    toSq: mv.toSq,
    dropPt: mv.dropPt,
    promote: mv.promote,
    movedPiece: 0,
    capturedPiece: 0,
    addedHandIdx: -1, // if capture adds to hand
    prevSide: sideSign,
    prevHash: hashKey,
  };

  const sIdx = sideIdxFromSign(sideSign);

  if (mv.dropPt){
    // drop from hand
    const hidx = HAND_IDX.get(mv.dropPt);
    const oldC = handsA[sIdx][hidx];
    const newC = oldC - 1;
    handsA[sIdx][hidx] = newC;
    xorHandCount(sideSign, hidx, oldC, newC);

    // place on board
    const toSq = mv.toSq;
    // empty assumed
    boardA[toSq] = sideSign * mv.dropPt;
    hashKey ^= Z_PIECE[sIdx][mv.dropPt][toSq];

    // flip side
    sideSign = -sideSign;
    hashKey ^= Z_SIDE;

    undoStack.push(u);
    return;
  }

  const fromSq = mv.fromSq;
  const toSq = mv.toSq;
  const moved = boardA[fromSq];
  const cap = boardA[toSq];

  u.movedPiece = moved;
  u.capturedPiece = cap;

  // remove moved from fromSq
  {
    const pt = Math.abs(moved);
    const s = moved > 0 ? 0 : 1;
    hashKey ^= Z_PIECE[s][pt][fromSq];
  }
  boardA[fromSq] = 0;

  // capture
  if (cap){
    const capPt = Math.abs(cap);
    const capSide = cap > 0 ? 0 : 1;
    hashKey ^= Z_PIECE[capSide][capPt][toSq];

    const base = baseTypeOf(capPt);
    if (base !== PTYPE.OU) {
      const hidx = HAND_IDX.get(base);
      const oldC = handsA[sIdx][hidx];
      const newC = oldC + 1;
      handsA[sIdx][hidx] = newC;
      xorHandCount(sideSign, hidx, oldC, newC);
      u.addedHandIdx = hidx;
    }
  }

  // place moved to toSq (maybe promoted)
  let newPt = Math.abs(moved);
  if (mv.promote){
    const p = PROM[newPt];
    if (p) newPt = p;
  }
  boardA[toSq] = sideSign * newPt;
  hashKey ^= Z_PIECE[sIdx][newPt][toSq];

  // flip side
  sideSign = -sideSign;
  hashKey ^= Z_SIDE;

  undoStack.push(u);
}

function unmakeMove(){
  const u = undoStack.pop();
  if (!u) return;

  // restore whole hash fast
  sideSign = u.prevSide;
  hashKey = u.prevHash;

  if (u.dropPt){
    // remove dropped piece from board
    boardA[u.toSq] = 0;

    // restore hand count (+1 back)
    const sIdx = sideIdxFromSign(sideSign);
    const hidx = HAND_IDX.get(u.dropPt);
    handsA[sIdx][hidx] += 1;
    return;
  }

  // restore pieces
  boardA[u.fromSq] = u.movedPiece;
  boardA[u.toSq] = u.capturedPiece;

  // restore hands if capture happened
  if (u.capturedPiece && u.addedHandIdx >= 0){
    const sIdx = sideIdxFromSign(sideSign);
    handsA[sIdx][u.addedHandIdx] -= 1;
  }
}

/* ------------------ Check / Attacks ------------------ */
function findKingSq(sign){
  const king = sign * PTYPE.OU;
  for (let sq=0;sq<81;sq++){
    if (boardA[sq] === king) return sq;
  }
  return -1;
}

function attacksSquare(fromSq, pAbs, pSign, targetSq){
  const fx = sqX(fromSq), fy = sqY(fromSq);
  const tx = sqX(targetSq), ty = sqY(targetSq);

  const dir = (pSign === 1) ? -1 : 1; // forward y

  const step = (dx,dy) => (fx+dx===tx && fy+dy===ty);
  const slide = (dx,dy) => {
    for (let k=1;k<9;k++){
      const nx=fx+dx*k, ny=fy+dy*k;
      if (!inside(nx,ny)) return false;
      const sq = sqIndex(nx,ny);
      if (sq === targetSq) return true;
      if (boardA[sq] !== 0) return false;
    }
    return false;
  };

  switch (pAbs){
    case PTYPE.FU: return step(0,dir);
    case PTYPE.KY: return slide(0,dir);
    case PTYPE.KE: return step(-1,2*dir) || step(1,2*dir);
    case PTYPE.GI:
      return step(-1,dir)||step(0,dir)||step(1,dir)||step(-1,-dir)||step(1,-dir);
    case PTYPE.KI:
    case PTYPE.TO: case PTYPE.NY: case PTYPE.NK: case PTYPE.NG:
      return step(-1,dir)||step(0,dir)||step(1,dir)||step(-1,0)||step(1,0)||step(0,-dir);
    case PTYPE.KA: return slide(1,1)||slide(1,-1)||slide(-1,1)||slide(-1,-1);
    case PTYPE.HI: return slide(1,0)||slide(-1,0)||slide(0,1)||slide(0,-1);
    case PTYPE.OU: {
      for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) {
        if (!dx && !dy) continue;
        if (step(dx,dy)) return true;
      }
      return false;
    }
    case PTYPE.UM:
      return (slide(1,1)||slide(1,-1)||slide(-1,1)||slide(-1,-1)) ||
             step(1,0)||step(-1,0)||step(0,1)||step(0,-1);
    case PTYPE.RY:
      return (slide(1,0)||slide(-1,0)||slide(0,1)||slide(0,-1)) ||
             step(1,1)||step(1,-1)||step(-1,1)||step(-1,-1);
  }
  return false;
}

function isKingInCheck(sign){
  const ksq = findKingSq(sign);
  if (ksq < 0) return false;
  const foe = -sign;

  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    if ((p > 0 ? 1 : -1) !== foe) continue;
    if (attacksSquare(sq, Math.abs(p), foe, ksq)) return true;
  }
  return false;
}

/* ------------------ Move generation (legal) ------------------ */
function hasUnpromotedPawnOnFile(sign, fileX){
  const pawn = sign * PTYPE.FU;
  for (let y=0;y<9;y++){
    const sq = sqIndex(fileX,y);
    if (boardA[sq] === pawn) return true;
  }
  return false;
}

function pushMoveWithPromo(moves, fromSq, toSq, ptAbs, promotePossible, forced){
  if (!promotePossible){
    moves.push({fromSq, toSq, dropPt:0, promote:0, movedPt:ptAbs, capturedPiece:boardA[toSq], givesCheck:false, key:0});
    return;
  }
  if (forced){
    moves.push({fromSq, toSq, dropPt:0, promote:1, movedPt:ptAbs, capturedPiece:boardA[toSq], givesCheck:false, key:0});
    return;
  }
  // both
  moves.push({fromSq, toSq, dropPt:0, promote:0, movedPt:ptAbs, capturedPiece:boardA[toSq], givesCheck:false, key:0});
  moves.push({fromSq, toSq, dropPt:0, promote:1, movedPt:ptAbs, capturedPiece:boardA[toSq], givesCheck:false, key:0});
}

function genPseudoMovesForPiece(moves, fromSq, pAbs, pSign){
  const x = sqX(fromSq), y = sqY(fromSq);
  const dir = (pSign === 1) ? -1 : 1;

  const addStep = (dx,dy) => {
    const nx = x+dx, ny = y+dy;
    if (!inside(nx,ny)) return;
    const toSq = sqIndex(nx,ny);
    const tp = boardA[toSq];
    if (tp && (tp > 0 ? 1 : -1) === pSign) return;

    const promoPossible = canPromote(pAbs) && (inPromoZone(pSign, y) || inPromoZone(pSign, ny));
    const forced = mustPromote(pAbs, pSign, ny);
    pushMoveWithPromo(moves, fromSq, toSq, pAbs, promoPossible, forced);
  };

  const addSlide = (dx,dy) => {
    for (let k=1;k<9;k++){
      const nx=x+dx*k, ny=y+dy*k;
      if (!inside(nx,ny)) break;
      const toSq = sqIndex(nx,ny);
      const tp = boardA[toSq];
      if (tp && (tp > 0 ? 1 : -1) === pSign) break;

      const promoPossible = canPromote(pAbs) && (inPromoZone(pSign, y) || inPromoZone(pSign, ny));
      const forced = mustPromote(pAbs, pSign, ny);
      pushMoveWithPromo(moves, fromSq, toSq, pAbs, promoPossible, forced);

      if (tp) break;
    }
  };

  switch (pAbs){
    case PTYPE.FU: addStep(0,dir); break;
    case PTYPE.KY: addSlide(0,dir); break;
    case PTYPE.KE: addStep(-1,2*dir); addStep(1,2*dir); break;
    case PTYPE.GI:
      addStep(-1,dir); addStep(0,dir); addStep(1,dir);
      addStep(-1,-dir); addStep(1,-dir);
      break;
    case PTYPE.KI:
    case PTYPE.TO: case PTYPE.NY: case PTYPE.NK: case PTYPE.NG:
      addStep(-1,dir); addStep(0,dir); addStep(1,dir);
      addStep(-1,0); addStep(1,0);
      addStep(0,-dir);
      break;
    case PTYPE.KA: addSlide(1,1); addSlide(1,-1); addSlide(-1,1); addSlide(-1,-1); break;
    case PTYPE.HI: addSlide(1,0); addSlide(-1,0); addSlide(0,1); addSlide(0,-1); break;
    case PTYPE.OU:
      for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) if (dx||dy) addStep(dx,dy);
      break;
    case PTYPE.UM:
      addSlide(1,1); addSlide(1,-1); addSlide(-1,1); addSlide(-1,-1);
      addStep(1,0); addStep(-1,0); addStep(0,1); addStep(0,-1);
      break;
    case PTYPE.RY:
      addSlide(1,0); addSlide(-1,0); addSlide(0,1); addSlide(0,-1);
      addStep(1,1); addStep(1,-1); addStep(-1,1); addStep(-1,-1);
      break;
  }
}

function genDropMoves(moves, sign){
  const sIdx = sideIdxFromSign(sign);
  for (let i=0;i<7;i++){
    const pt = HAND_ORDER[i];
    if (handsA[sIdx][i] <= 0) continue;

    for (let sq=0;sq<81;sq++){
      if (boardA[sq]) continue;
      const x = sqX(sq), y = sqY(sq);

      // cannot drop on last ranks
      if ((pt===PTYPE.FU || pt===PTYPE.KY) && ((sign===1 && y===0) || (sign===-1 && y===8))) continue;
      if (pt===PTYPE.KE && ((sign===1 && y<=1) || (sign===-1 && y>=7))) continue;

      // nifu
      if (pt===PTYPE.FU && hasUnpromotedPawnOnFile(sign, x)) continue;

      moves.push({fromSq:-1, toSq:sq, dropPt:pt, promote:0, movedPt:pt, capturedPiece:0, givesCheck:false, key:0});
    }
  }
}

function hasAnyLegalMoveCurrentSide(){
  const sign = sideSign;

  // pseudo list, stop at first legal
  const pseudo = [];
  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    const ps = p > 0 ? 1 : -1;
    if (ps !== sign) continue;
    genPseudoMovesForPiece(pseudo, sq, Math.abs(p), ps);
    if (pseudo.length > 128) break; // enough; still safe
  }
  genDropMoves(pseudo, sign);

  for (let i=0;i<pseudo.length;i++){
    const mv = pseudo[i];
    makeMove(mv);
    const ok = !isKingInCheck(-sideSign); // after makeMove, sideSign flipped. need original sign -> now -sideSign
    if (ok){
      unmakeMove();
      return true;
    }
    unmakeMove();
  }
  return false;
}

function generateLegalMovesForCurrentSide() {
  const sign = sideSign;
  const pseudo = [];
  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    const ps = p > 0 ? 1 : -1;
    if (ps !== sign) continue;
    genPseudoMovesForPiece(pseudo, sq, Math.abs(p), ps);
  }
  genDropMoves(pseudo, sign);

  const legals = [];
  for (let i=0;i<pseudo.length;i++){
    const mv = pseudo[i];

    makeMove(mv);
    // after makeMove, sideSign flipped (opponent to move).
    // legality: our king (original sign) must not be in check
    const illegal = isKingInCheck(-sideSign); // original sign = -sideSign
    if (illegal){
      unmakeMove();
      continue;
    }

    // givesCheck?
    const gives = isKingInCheck(sideSign); // opponent king in check? (opponent sign = sideSign)
    mv.givesCheck = gives;

    // uchi-fuzume simple (pawn drop checkmate)
    if (mv.dropPt === PTYPE.FU && gives){
      // if opponent has no legal move => illegal
      const any = hasAnyLegalMoveCurrentSide(); // current side is opponent now (sideSign)
      if (!any){
        unmakeMove();
        continue;
      }
    }

    unmakeMove();

    mv.key = moveKey(mv);
    legals.push(mv);
  }
  return legals;
}

/* ------------------ NN Encode (D=2283) ------------------ */
const NUM_PTYPE = 14;
const NUM_SQ = 81;
const D = 1 + (2 * NUM_PTYPE * NUM_SQ) + (2 * 7); // 2283

function encodeToFloat32() {
  const x = new Float32Array(D);
  x[0] = (sideSign === 1) ? 1.0 : 0.0;

  const base = 1;
  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    const color = (p > 0) ? 0 : 1; // S=0, G=1
    const pidx = Math.abs(p) - 1; // 0..13
    const idx = base + (color * NUM_PTYPE + pidx) * NUM_SQ + sq;
    x[idx] = 1.0;
  }

  const base2 = 1 + (2 * NUM_PTYPE * NUM_SQ);
  // S hands then G hands
  for (let i=0;i<7;i++){
    x[base2 + i] = handsA[0][i];
    x[base2 + 7 + i] = handsA[1][i];
  }
  return x;
}

/* ------------------ Eval (NN / fallback) + cache ------------------ */
const evalCache = new Map(); // hashKey(BigInt) -> score (side-to-move perspective)
const EVAL_CACHE_MAX = 50000;

function materialFallbackScore(){
  // score from side-to-move perspective
  let s = 0;
  for (let sq=0;sq<81;sq++){
    const p = boardA[sq];
    if (!p) continue;
    const pt = Math.abs(p);
    const v = VALUE[pt] || 0;
    s += (p > 0) ? v : -v; // S view
  }
  for (let i=0;i<7;i++){
    s += handsA[0][i] * (VALUE[HAND_ORDER[i]]||0);
    s -= handsA[1][i] * (VALUE[HAND_ORDER[i]]||0);
  }
  // convert to side-to-move view
  return (sideSign === 1) ? s : -s;
}

function evalScore(useNN){
  const key = hashKey;
  const cached = evalCache.get(key);
  if (cached != null) return cached;

  let score = 0;
  if (useNN && tfReady && model) {
    try {
      const feats = encodeToFloat32();
      // model output is assumed to be "side-to-move" value in [-1,1]
      let v = 0;
      tf.tidy(() => {
        const t = tf.tensor(feats, [1, D], "float32");
        const out = model.predict(t);
        v = out.dataSync()[0];
      });
      score = v * 10000.0;
    } catch (e) {
      score = materialFallbackScore();
    }
  } else {
    score = materialFallbackScore();
  }

  // cache bound
  if (evalCache.size > EVAL_CACHE_MAX) evalCache.clear();
  evalCache.set(key, score);
  return score;
}

/* ------------------ TT (Transposition Table) ------------------ */
const TT = new Map(); // BigInt -> entry
const TT_MAX = 200000;

const TT_EXACT = 0;
const TT_LOWER = 1;
const TT_UPPER = 2;

let searchGen = 0;

function ttGet(key){
  return TT.get(key);
}
function ttPut(key, entry){
  if (TT.size > TT_MAX) TT.clear();
  TT.set(key, entry);
}

/* ------------------ Move ordering heuristics ------------------ */
const killer1 = new Uint32Array(128);
const killer2 = new Uint32Array(128);
// history[sideIdx][from*81+to] : bonus
const historyH = [new Int32Array(81*81), new Int32Array(81*81)];

function isCapture(mv){ return mv.capturedPiece !== 0; }

function orderScoreForMove(mv, ply, ttMoveKey){
  let s = 0;

  // TT move first
  if (ttMoveKey !== 0 && sameMoveKey(mv.key, ttMoveKey)) s += 1_000_000_000;

  // captures: MVV-LVA style
  if (mv.capturedPiece){
    const capPt = Math.abs(mv.capturedPiece);
    const mvPt = mv.movedPt;
    s += 200_000 + (VALUE[capPt]||0) * 10 - (VALUE[mvPt]||0);
  }

  // promote
  if (!mv.dropPt && mv.promote) s += 20_000;

  // gives check
  if (mv.givesCheck) s += 15_000;

  // killers (quiet moves)
  if (!mv.capturedPiece && !mv.dropPt) {
    if (mv.key === killer1[ply]) s += 12_000;
    else if (mv.key === killer2[ply]) s += 8_000;

    // history
    const sIdx = sideIdxFromSign(-sideSign); // careful: during ordering, sideSign is current. we are scoring for current side.
    const from = mv.fromSq >= 0 ? mv.fromSq : mv.toSq; // drop uses toSq
    const to = mv.toSq;
    s += historyH[sIdx][from*81 + to] | 0;
  }

  return s;
}

function updateKillersAndHistory(mv, ply, depth){
  // update on beta cutoff
  if (mv.capturedPiece || mv.dropPt) return; // quiet only for killer/history
  const key = mv.key;

  if (killer1[ply] !== key){
    killer2[ply] = killer1[ply];
    killer1[ply] = key;
  }

  const sIdx = sideIdxFromSign(-sideSign); // current side before makeMove
  const from = mv.fromSq;
  const to = mv.toSq;
  if (from >= 0){
    const idx = from*81 + to;
    historyH[sIdx][idx] += depth * depth;
    if (historyH[sIdx][idx] > 1_000_000) historyH[sIdx].fill(0); // prevent overflow
  }
}

/* ------------------ Search (alpha-beta + TT + aspiration) ------------------ */
let stop = false;
let tStart = 0;
let tLimit = 0;
let nodes = 0;

function timeUp(){
  return (performance.now() - tStart) >= tLimit;
}

const INF = 1e15;
const MATE = 1_000_000;

function negamax(depth, alpha, beta, ply, useNN){
  if (stop) return 0;
  if ((nodes & 2047) === 0 && timeUp()){
    stop = true;
    return 0;
  }
  nodes++;

  const alpha0 = alpha;

  // TT probe
  const key = hashKey;
  const ent = ttGet(key);
  let ttMoveKey = 0;
  if (ent){
    ttMoveKey = ent.bestKey || 0;
    if (ent.depth >= depth){
      const v = ent.value;
      if (ent.flag === TT_EXACT) return v;
      if (ent.flag === TT_LOWER) alpha = Math.max(alpha, v);
      else if (ent.flag === TT_UPPER) beta = Math.min(beta, v);
      if (alpha >= beta) return v;
    }
  }

  // generate legal
  const legals = generateLegalMovesForCurrentSide();
  if (legals.length === 0){
    // checkmate/stalemate
    if (isKingInCheck(sideSign)) return -MATE + ply;
    return 0;
  }

  if (depth <= 0){
    return evalScore(useNN);
  }

  // order moves
  for (let i=0;i<legals.length;i++){
    legals[i]._ord = orderScoreForMove(legals[i], ply, ttMoveKey);
  }
  legals.sort((a,b)=> (b._ord - a._ord));

  let bestVal = -INF;
  let bestKey = 0;

  for (let i=0;i<legals.length;i++){
    const mv = legals[i];
    makeMove(mv);
    const v = -negamax(depth-1, -beta, -alpha, ply+1, useNN);
    unmakeMove();

    if (stop) break;

    if (v > bestVal){
      bestVal = v;
      bestKey = mv.key;
    }

    if (v > alpha) alpha = v;
    if (alpha >= beta){
      // beta cutoff
      updateKillersAndHistory(mv, ply, depth);
      break;
    }
  }

  // TT store
  let flag = TT_EXACT;
  if (bestVal <= alpha0) flag = TT_UPPER;
  else if (bestVal >= beta) flag = TT_LOWER;

  ttPut(key, {
    depth,
    flag,
    value: bestVal,
    bestKey,
    gen: searchGen,
  });

  return bestVal;
}

function searchRoot(depth, alpha, beta, useNN){
  const legals = generateLegalMovesForCurrentSide();
  if (legals.length === 0){
    if (isKingInCheck(sideSign)) return {score: -MATE, best:null};
    return {score: 0, best:null};
  }

  // TT best at root
  const ent = ttGet(hashKey);
  const ttMoveKey = ent?.bestKey || 0;

  for (let i=0;i<legals.length;i++){
    legals[i]._ord = orderScoreForMove(legals[i], 0, ttMoveKey);
  }
  legals.sort((a,b)=> (b._ord - a._ord));

  let best = null;
  let bestScore = -INF;

  for (let i=0;i<legals.length;i++){
    const mv = legals[i];

    makeMove(mv);
    const v = -negamax(depth-1, -beta, -alpha, 1, useNN);
    unmakeMove();

    if (stop) break;

    if (v > bestScore){
      bestScore = v;
      best = mv;
    }
    if (v > alpha) alpha = v;
  }

  return {score: bestScore, best};
}

/* aspiration window iterative deepening */
function thinkIterative(depthMax, timeMs, useNN, requestId){
  stop = false;
  nodes = 0;
  tStart = performance.now();
  tLimit = Math.max(20, timeMs|0);

  searchGen++;

  let bestMove = null;
  let bestScore = 0;
  let depthDone = 0;

  // aspiration params
  let prev = 0;
  let window = 250; // initial aspiration window

  for (let d=1; d<=depthMax; d++){
    if (timeUp()){ stop = true; break; }

    let a = -INF, b = INF;

    if (d >= 2){
      a = prev - window;
      b = prev + window;
    }

    // aspiration loop
    let result;
    while (true){
      result = searchRoot(d, a, b, useNN);
      if (stop) break;

      const s = result.score;

      if (d < 2) break;

      if (s <= a){
        // fail-low
        a = -INF;
        window *= 2;
        b = prev + window;
        continue;
      }
      if (s >= b){
        // fail-high
        b = INF;
        window *= 2;
        a = prev - window;
        continue;
      }
      break;
    }

    if (stop) break;

    if (result.best){
      bestMove = result.best;
      bestScore = result.score;
      prev = bestScore;
      depthDone = d;

      postMessage({
        type: "progress",
        requestId,
        depthDone,
        depthMax,
        bestMove: mvToUI(bestMove),
        bestStr: mvToBestStr(bestMove),
        bestScore,
        partial: true,
      });
    } else {
      // no best move
      break;
    }
  }

  const partial = (depthDone < depthMax);
  const elapsed = (performance.now() - tStart) | 0;

  return {
    ok: !!bestMove,
    depthDone,
    depthMax,
    bestMove,
    bestScore,
    partial,
    timeMs: elapsed,
    nodes,
  };
}

/* ------------------ Load position from main ------------------ */
function loadPos(pos){
  // board
  boardA.fill(0);
  handsA[0].fill(0);
  handsA[1].fill(0);

  // side
  sideSign = (pos.sideToMove === "S") ? 1 : -1;

  // board[y][x]
  for (let y=0;y<9;y++){
    for (let x=0;x<9;x++){
      const p = pos.board?.[y]?.[x];
      if (!p) continue;
      const pt = PTYPE[p.type] || 0;
      if (!pt) continue;
      const s = (p.side === "S") ? 1 : -1;
      const sq = sqIndex(x,y);
      boardA[sq] = s * pt;
    }
  }

  // hands
  const hs = pos.hands || {};
  const hS = hs.S || {};
  const hG = hs.G || {};
  const fillHand = (arr, hObj) => {
    arr[0] = (hObj.FU|0) || 0;
    arr[1] = (hObj.KY|0) || 0;
    arr[2] = (hObj.KE|0) || 0;
    arr[3] = (hObj.GI|0) || 0;
    arr[4] = (hObj.KI|0) || 0;
    arr[5] = (hObj.KA|0) || 0;
    arr[6] = (hObj.HI|0) || 0;
  };
  fillHand(handsA[0], hS);
  fillHand(handsA[1], hG);

  // hash
  hashKey = computeHash();
  undoStack.length = 0;
}

/* ------------------ Worker message handling ------------------ */
self.onmessage = async (e) => {
  const msg = e.data || {};

  if (msg.type === "init") {
    const modelUrl = msg.modelUrl || "./shogi_eval_wars_tfjs/model.json";
    const backend = msg.backend || "wasm";

    const r = await initTf(modelUrl, backend);
    if (r.ok) {
      postMessage({ type:"init_done", ok:true, info: modelInfo });
    } else {
      postMessage({ type:"init_done", ok:false, error: r.error || "init failed" });
    }
    return;
  }

  if (msg.type === "think") {
    const requestId = msg.requestId || 0;
    const depthMax = Math.max(1, Math.min(10, msg.depthMax|0));
    const timeMs = Math.max(50, msg.timeMs|0);
    const useNN = !!msg.useNN;

    // (optional) clear evalCache each think to avoid memory growth / stale
    evalCache.clear();

    loadPos(msg.pos);

    const r = thinkIterative(depthMax, timeMs, useNN, requestId);

    postMessage({
      type: "result",
      requestId,
      ok: r.ok,
      depthDone: r.depthDone,
      depthMax: r.depthMax,
      bestMove: r.bestMove ? mvToUI(r.bestMove) : null,
      bestStr: r.bestMove ? mvToBestStr(r.bestMove) : "-",
      bestScore: r.bestScore,
      partial: r.partial,
      timeMs: r.timeMs,
      nodes: r.nodes,
    });
    return;
  }
};
