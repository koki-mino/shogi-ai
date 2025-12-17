/* worker.js v5
   - 盤面 clone を廃止（make/unmake）
   - 内部表現: board[81] int (0 empty, +ptype S, -ptype G)
   - 反復深化 + αβ + 時間打ち切り（UI非ブロック）
   - NN評価は葉で使用（TFJS GraphModel）
*/

let tf = null;
let model = null;
let modelStatus = "未読込";

const NUM_PTYPE = 14, NUM_SQ = 81;
const HAND_ORDER = ['FU','KY','KE','GI','KI','KA','HI'];
const D = 1 + (2 * NUM_PTYPE * NUM_SQ) + (2 * HAND_ORDER.length);

const VALUE = {FU:100,KY:300,KE:300,GI:400,KI:500,KA:700,HI:800,OU:0,TO:500,NY:500,NK:500,NG:500,UM:900,RY:1000};

const PROM_MAP = {FU:"TO", KY:"NY", KE:"NK", GI:"NG", KA:"UM", HI:"RY"};
const UNPROM_MAP = {TO:"FU", NY:"KY", NK:"KE", NG:"GI", UM:"KA", RY:"HI"};

const PTYPE_ID = {
  FU: 1, KY: 2, KE: 3, GI: 4, KI: 5, KA: 6, HI: 7,
  OU: 8, TO: 9, NY: 10, NK: 11, NG: 12, UM: 13, RY: 14,
};
const ID_PTYPE = Object.fromEntries(Object.entries(PTYPE_ID).map(([k,v])=>[v,k]));

// python-shogi と一致（答え合わせ済み）
function sqIndex(x, y) { return y*9 + x; }
function xyFromSq(sq){ return {x: sq % 9, y: (sq/9)|0}; }

let inputKey = "input_layer_5";
let outputName = "Identity:0";
let expectedD = 2283;

let cancelRequestId = -1;

/* ------------------ TFJS ------------------ */
function encodeForNN(board81, handsS, handsG, sideToMove /*0=S,1=G*/){
  const x = new Float32Array(D);
  x[0] = (sideToMove === 0) ? 1.0 : 0.0;

  const base = 1;
  for (let sq=0; sq<81; sq++){
    const pc = board81[sq];
    if (!pc) continue;
    const color = (pc > 0) ? 0 : 1;
    const pt = Math.abs(pc);
    const pidx = pt - 1;
    const idx = base + ((color * NUM_PTYPE + pidx) * NUM_SQ + sq);
    x[idx] = 1.0;
  }

  const base2 = 1 + (2 * NUM_PTYPE * NUM_SQ);
  for (let i=0;i<HAND_ORDER.length;i++) x[base2+i] = handsS[i] || 0;
  for (let i=0;i<HAND_ORDER.length;i++) x[base2+HAND_ORDER.length+i] = handsG[i] || 0;
  return x;
}

function predictSenteValue(feats2283){
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

/* ------------------ Internal representation ------------------ */
// side: 0=S, 1=G
function opp(side){ return side ^ 1; }

function typeNameFromAbsId(absId){ return ID_PTYPE[absId] || "FU"; }
function absIdFromTypeName(name){ return PTYPE_ID[name] || 1; }

function handIndexOfTypeName(t){
  // FU,KY,KE,GI,KI,KA,HI only
  const i = HAND_ORDER.indexOf(t);
  return i;
}
function addHand(hands, tName, delta){
  const i = handIndexOfTypeName(tName);
  if (i >= 0) hands[i] += delta;
}

function unpromoteTypeName(tName){
  return UNPROM_MAP[tName] || tName;
}
function promoteTypeName(tName){
  return PROM_MAP[tName] || tName;
}

function inPromoZone(side, y){ // side 0=S: y<=2, side1=G: y>=6
  return side===0 ? (y<=2) : (y>=6);
}
function canPromote(tName){
  return !!PROM_MAP[tName];
}
function mustPromote(tName, side, toY){
  if (tName === 'FU' || tName === 'KY') return side===0 ? (toY===0) : (toY===8);
  if (tName === 'KE') return side===0 ? (toY<=1) : (toY>=7);
  return false;
}

/* ------------------ Convert UI pos -> internal ------------------ */
function convertPos(pos){
  // pos.board[y][x] = {side:'S'|'G', type:'FU'..}
  const board81 = new Int16Array(81);
  let kingSqS = -1, kingSqG = -1;

  for (let y=0;y<9;y++) for (let x=0;x<9;x++){
    const p = pos.board[y][x];
    if (!p) continue;
    const absId = absIdFromTypeName(p.type);
    const sq = sqIndex(x,y);
    const pc = (p.side === 'S') ? absId : -absId;
    board81[sq] = pc;
    if (absId === PTYPE_ID.OU){
      if (p.side === 'S') kingSqS = sq;
      else kingSqG = sq;
    }
  }

  const handsS = new Int16Array(HAND_ORDER.length);
  const handsG = new Int16Array(HAND_ORDER.length);
  for (let i=0;i<HAND_ORDER.length;i++){
    const t = HAND_ORDER[i];
    handsS[i] = (pos.hands.S[t] || 0);
    handsG[i] = (pos.hands.G[t] || 0);
  }

  const side = (pos.sideToMove === 'S') ? 0 : 1;

  return { board81, handsS, handsG, side, kingSqS, kingSqG };
}

/* ------------------ Move generation (make/unmake) ------------------ */
// move internal form:
// { fromSq: number|-1, toSq: number, dropAbsId?: number, promote?: boolean, capPc?: number, movedPc?: number, prevKingS?:number, prevKingG?:number, handDelta?: [side, idx, delta] ... }
function isOwnPiece(pc, side){ return side===0 ? (pc>0) : (pc<0); }
function isFoePiece(pc, side){ return side===0 ? (pc<0) : (pc>0); }

function hasUnpromotedPawnOnFile(board81, side, fileX){
  const pawnId = PTYPE_ID.FU; // unpromoted pawn
  for (let y=0;y<9;y++){
    const sq = sqIndex(fileX,y);
    const pc = board81[sq];
    if (!pc) continue;
    if (side===0){
      if (pc === pawnId) return true;
    } else {
      if (pc === -pawnId) return true;
    }
  }
  return false;
}

function genPseudoMoves(board81, side, kingSqS, kingSqG, handsS, handsG){
  const moves = [];
  const forward = (side===0) ? -1 : 1;

  function pushMove(fromSq, toSq, promote){
    moves.push({fromSq, toSq, promote: !!promote});
  }

  function pushMoveWithPromo(fromSq, toSq, tName, fromY, toY){
    if (!canPromote(tName)){
      pushMove(fromSq,toSq,false); return;
    }
    const promoPossible = inPromoZone(side, fromY) || inPromoZone(side, toY);
    const forced = mustPromote(tName, side, toY);
    if (!promoPossible){
      pushMove(fromSq,toSq,false); return;
    }
    if (forced){
      pushMove(fromSq,toSq,true); return;
    }
    pushMove(fromSq,toSq,false);
    pushMove(fromSq,toSq,true);
  }

  // board moves
  for (let sq=0;sq<81;sq++){
    const pc = board81[sq];
    if (!pc) continue;
    if (!isOwnPiece(pc, side)) continue;

    const absId = Math.abs(pc);
    const tName = typeNameFromAbsId(absId);
    const {x,y} = xyFromSq(sq);

    // helpers
    const addStep = (dx,dy)=>{
      const nx = x+dx, ny=y+dy;
      if (nx<0||nx>=9||ny<0||ny>=9) return;
      const toSq = sqIndex(nx,ny);
      const tp = board81[toSq];
      if (tp && isOwnPiece(tp, side)) return;
      pushMoveWithPromo(sq, toSq, tName, y, ny);
    };
    const addSlide = (dx,dy)=>{
      for (let k=1;k<9;k++){
        const nx=x+dx*k, ny=y+dy*k;
        if (nx<0||nx>=9||ny<0||ny>=9) break;
        const toSq=sqIndex(nx,ny);
        const tp=board81[toSq];
        if (tp && isOwnPiece(tp, side)) break;
        pushMoveWithPromo(sq,toSq,tName,y,ny);
        if (tp) break;
      }
    };

    // piece
    switch(tName){
      case 'FU': addStep(0, forward); break;
      case 'KY': addSlide(0, forward); break;
      case 'KE': addStep(-1, 2*forward); addStep(1, 2*forward); break;
      case 'GI':
        addStep(-1, forward); addStep(0, forward); addStep(1, forward);
        addStep(-1, -forward); addStep(1, -forward);
        break;
      case 'KI':
      case 'TO': case 'NY': case 'NK': case 'NG':
        addStep(-1, forward); addStep(0, forward); addStep(1, forward);
        addStep(-1, 0); addStep(1, 0);
        addStep(0, -forward);
        break;
      case 'KA': addSlide(1,1); addSlide(1,-1); addSlide(-1,1); addSlide(-1,-1); break;
      case 'HI': addSlide(1,0); addSlide(-1,0); addSlide(0,1); addSlide(0,-1); break;
      case 'OU':
        for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) if(dx||dy) addStep(dx,dy);
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
  }

  // drops
  const hands = (side===0) ? handsS : handsG;
  for (let i=0;i<HAND_ORDER.length;i++){
    const n = hands[i];
    if (n<=0) continue;
    const tName = HAND_ORDER[i];

    for (let y=0;y<9;y++) for (let x=0;x<9;x++){
      const toSq = sqIndex(x,y);
      if (board81[toSq]) continue;

      // rank restrictions
      if (tName==='FU' || tName==='KY'){
        if ((side===0 && y===0) || (side===1 && y===8)) continue;
      }
      if (tName==='KE'){
        if ((side===0 && y<=1) || (side===1 && y>=7)) continue;
      }
      // nifu
      if (tName==='FU'){
        if (hasUnpromotedPawnOnFile(board81, side, x)) continue;
      }

      moves.push({fromSq:-1, toSq, dropAbsId: absIdFromTypeName(tName), promote:false});
    }
  }

  return moves;
}

/* ------------------ Check detection ------------------ */
function attacksSquare(board81, fromSq, pc, tx, ty){
  const side = (pc>0) ? 0 : 1;
  const absId = Math.abs(pc);
  const tName = typeNameFromAbsId(absId);
  const {x,y} = xyFromSq(fromSq);
  const forward = (side===0) ? -1 : 1;

  const step = (dx,dy)=> (x+dx===tx && y+dy===ty);
  const slide = (dx,dy)=>{
    for (let k=1;k<9;k++){
      const nx=x+dx*k, ny=y+dy*k;
      if (nx<0||nx>=9||ny<0||ny>=9) return false;
      const sq = sqIndex(nx,ny);
      if (nx===tx && ny===ty) return true;
      if (board81[sq]) return false;
    }
    return false;
  };

  switch(tName){
    case 'FU': return step(0, forward);
    case 'KY': return slide(0, forward);
    case 'KE': return step(-1,2*forward) || step(1,2*forward);
    case 'GI':
      return step(-1,forward)||step(0,forward)||step(1,forward)||step(-1,-forward)||step(1,-forward);
    case 'KI':
    case 'TO': case 'NY': case 'NK': case 'NG':
      return step(-1,forward)||step(0,forward)||step(1,forward)||step(-1,0)||step(1,0)||step(0,-forward);
    case 'KA': return slide(1,1)||slide(1,-1)||slide(-1,1)||slide(-1,-1);
    case 'HI': return slide(1,0)||slide(-1,0)||slide(0,1)||slide(0,-1);
    case 'OU':
      for (const dx of [-1,0,1]) for (const dy of [-1,0,1]) if(dx||dy) if(step(dx,dy)) return true;
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

function isKingInCheck(board81, side, kingSqS, kingSqG){
  const ksq = (side===0) ? kingSqS : kingSqG;
  if (ksq < 0) return false;
  const {x:kx, y:ky} = xyFromSq(ksq);
  const foeSide = opp(side);

  for (let sq=0;sq<81;sq++){
    const pc = board81[sq];
    if (!pc) continue;
    if (foeSide===0 && pc<=0) continue;
    if (foeSide===1 && pc>=0) continue;
    if (attacksSquare(board81, sq, pc, kx, ky)) return true;
  }
  return false;
}

/* ------------------ make/unmake ------------------ */
function makeMove(state, mv, stack){
  // state: {board81,handsS,handsG,side,kingSqS,kingSqG}
  const {board81} = state;
  const side = state.side;

  const rec = {
    fromSq: mv.fromSq,
    toSq: mv.toSq,
    dropAbsId: mv.dropAbsId || 0,
    promote: !!mv.promote,
    capPc: board81[mv.toSq] || 0,
    movedPc: 0,
    prevSide: side,
    prevKingS: state.kingSqS,
    prevKingG: state.kingSqG,
    handIncType: 0, // captured base type absId (0 if none)
  };

  if (mv.fromSq === -1) {
    // drop
    const absId = mv.dropAbsId;
    const pc = (side===0) ? absId : -absId;
    board81[mv.toSq] = pc;
    // hand--
    const hands = (side===0) ? state.handsS : state.handsG;
    const tName = typeNameFromAbsId(absId);
    const hi = handIndexOfTypeName(tName);
    if (hi>=0) hands[hi]--;

    rec.movedPc = pc;
  } else {
    const fromPc = board81[mv.fromSq];
    rec.movedPc = fromPc;

    // capture -> hand add base
    if (rec.capPc) {
      const capAbs = Math.abs(rec.capPc);
      const capName = typeNameFromAbsId(capAbs);
      const baseName = unpromoteTypeName(capName);
      const baseAbs = absIdFromTypeName(baseName);
      rec.handIncType = baseAbs;

      // OUは持ち駒にしない（そもそも入らないが念のため）
      if (baseAbs !== PTYPE_ID.OU) {
        const hands = (side===0) ? state.handsS : state.handsG;
        const hi = handIndexOfTypeName(baseName);
        if (hi>=0) hands[hi] += 1;
      }
    }

    board81[mv.fromSq] = 0;

    let absId = Math.abs(fromPc);
    let tName = typeNameFromAbsId(absId);
    if (mv.promote) tName = promoteTypeName(tName);
    absId = absIdFromTypeName(tName);

    const toPc = (side===0) ? absId : -absId;
    board81[mv.toSq] = toPc;

    // update king
    if (absId === PTYPE_ID.OU) {
      if (side===0) state.kingSqS = mv.toSq;
      else state.kingSqG = mv.toSq;
    }
  }

  state.side = opp(side);
  stack.push(rec);
  return rec;
}

function unmakeMove(state, stack){
  const rec = stack.pop();
  if (!rec) return;

  const {board81} = state;

  state.side = rec.prevSide;
  state.kingSqS = rec.prevKingS;
  state.kingSqG = rec.prevKingG;

  const side = rec.prevSide;

  if (rec.fromSq === -1) {
    // undo drop
    board81[rec.toSq] = 0;
    const absId = rec.dropAbsId;
    const tName = typeNameFromAbsId(absId);
    const hands = (side===0) ? state.handsS : state.handsG;
    const hi = handIndexOfTypeName(tName);
    if (hi>=0) hands[hi] += 1;
  } else {
    // undo move (restore from / restore capture)
    board81[rec.fromSq] = rec.movedPc;
    board81[rec.toSq] = rec.capPc;

    // undo hand add if captured
    if (rec.handIncType && rec.handIncType !== PTYPE_ID.OU) {
      const baseName = typeNameFromAbsId(rec.handIncType);
      const hands = (side===0) ? state.handsS : state.handsG;
      const hi = handIndexOfTypeName(baseName);
      if (hi>=0) hands[hi] -= 1;
    }
  }
}

/* ------------------ legality / uchi-fu-zume (simple) ------------------ */
function genLegalMoves(state){
  const pseudo = genPseudoMoves(state.board81, state.side, state.kingSqS, state.kingSqG, state.handsS, state.handsG);
  const legal = [];
  const stack = [];

  for (const mv of pseudo){
    makeMove(state, mv, stack);

    // self-check
    const ok = !isKingInCheck(state.board81, opp(state.side), state.kingSqS, state.kingSqG); // opp(state.side) == mover side
    if (ok) {
      // 打ち歩詰め（簡易）：歩打ちで王手 & 相手合法手0 なら禁止
      if (mv.fromSq === -1 && mv.dropAbsId === PTYPE_ID.FU) {
        const foe = state.side; // after move, side flipped -> foe is state.side
        if (isKingInCheck(state.board81, foe, state.kingSqS, state.kingSqG)) {
          const replies = genLegalMoves(state); // 再帰（重いが歩打ち時だけ）
          if (replies.length === 0) {
            unmakeMove(state, stack);
            continue;
          }
        }
      }
      legal.push(mv);
    }

    unmakeMove(state, stack);
  }

  return legal;
}

/* ------------------ eval ------------------ */
function materialEval(state){
  let s=0;
  const b=state.board81;

  for (let sq=0;sq<81;sq++){
    const pc=b[sq];
    if (!pc) continue;
    const absId=Math.abs(pc);
    const tName=typeNameFromAbsId(absId);
    const v = VALUE[tName] || 0;
    s += (pc>0) ? v : -v;
  }
  for (let i=0;i<HAND_ORDER.length;i++){
    const t = HAND_ORDER[i];
    const v = VALUE[t] || 0;
    s += v * (state.handsS[i]||0);
    s -= v * (state.handsG[i]||0);
  }
  return s; // sente view
}

function evalForSearch(state, useNN){
  if (!useNN || !model || !tf) {
    const m = materialEval(state);
    return (state.side===0) ? m : -m;
  }
  const feats = encodeForNN(state.board81, state.handsS, state.handsG, state.side);
  const v = predictSenteValue(feats);
  if (v == null || Number.isNaN(v)) {
    const m = materialEval(state);
    return (state.side===0) ? m : -m;
  }
  const senteScore = v * 10000.0;
  return (state.side===0) ? senteScore : -senteScore;
}

/* ------------------ search (iterative deepening + alpha-beta) ------------------ */
function timeOrCancel(deadline, requestId){
  if (requestId === cancelRequestId) throw new Error("__CANCEL__");
  if (Date.now() >= deadline) throw new Error("__TIME__");
}

function orderMovesSimple(state, moves, pvMove){
  // capture / promote / check 優先の簡易スコア
  const b = state.board81;
  const foe = state.side;

  const scored = moves.map(mv=>{
    let sc = 0;
    if (pvMove && mv.fromSq===pvMove.fromSq && mv.toSq===pvMove.toSq && (!!mv.promote)===(!!pvMove.promote) && (mv.dropAbsId||0)===(pvMove.dropAbsId||0)) {
      sc += 100000;
    }
    if (mv.fromSq !== -1) {
      const cap = b[mv.toSq];
      if (cap) {
        const capName = typeNameFromAbsId(Math.abs(cap));
        sc += (VALUE[capName]||0) + 200;
      }
      if (mv.promote) sc += 150;
    } else {
      sc += 20;
    }

    // check bonus (make/unmake one ply)
    const stack=[];
    makeMove(state,mv,stack);
    if (isKingInCheck(state.board81, foe, state.kingSqS, state.kingSqG)) sc += 400;
    unmakeMove(state,stack);

    return {mv, sc};
  });

  scored.sort((a,b)=>b.sc-a.sc);
  return scored.map(x=>x.mv);
}

function negamax(state, depth, alpha, beta, useNN, deadline, requestId, pvMove){
  timeOrCancel(deadline, requestId);

  const legals = genLegalMoves(state);
  if (legals.length===0){
    // side to move has no legal moves
    const inCheck = isKingInCheck(state.board81, state.side, state.kingSqS, state.kingSqG);
    return inCheck ? -999999 : 0;
  }
  if (depth<=0) return evalForSearch(state, useNN);

  const ordered = orderMovesSimple(state, legals, pvMove);
  const stack=[];

  let best = -Infinity;

  for (const mv of ordered){
    makeMove(state, mv, stack);
    const score = -negamax(state, depth-1, -beta, -alpha, useNN, deadline, requestId, null);
    unmakeMove(state, stack);

    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }
  return best;
}

function chooseBestRoot(state, depth, useNN, deadline, requestId, pvMove){
  timeOrCancel(deadline, requestId);

  const legals = genLegalMoves(state);
  if (legals.length===0) return null;

  const ordered = orderMovesSimple(state, legals, pvMove);
  const stack=[];

  let bestMove = ordered[0];
  let bestScore = -Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (const mv of ordered){
    makeMove(state, mv, stack);
    const score = -negamax(state, depth-1, -beta, -alpha, useNN, deadline, requestId, null);
    unmakeMove(state, stack);

    if (score > bestScore){
      bestScore = score;
      bestMove = mv;
    }
    if (score > alpha) alpha = score;
  }

  return {move: bestMove, score: bestScore};
}

function toUiMove(mv){
  // main.js の applyMove が期待する形式へ戻す
  if (mv.fromSq === -1){
    const to = xyFromSq(mv.toSq);
    return {from:null, to, drop: typeNameFromAbsId(mv.dropAbsId), promote:false};
  }
  const from = xyFromSq(mv.fromSq);
  const to = xyFromSq(mv.toSq);
  return {from, to, drop:null, promote: !!mv.promote};
}

/* ------------------ init TF ------------------ */
async function initTF(modelUrl, backend){
  try {
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

    model = await tf.loadGraphModel(modelUrl, { fromTFHub:false });

    // signature を読めたら key を更新
    try {
      const meta = await (await fetch(modelUrl, { cache:"no-store" })).json();
      const ins = meta?.signature?.inputs || null;
      const outs = meta?.signature?.outputs || null;
      const inKeys = ins ? Object.keys(ins) : [];
      const outKeys = outs ? Object.keys(outs) : [];
      if (inKeys.length) {
        inputKey = inKeys[0];
        expectedD = parseInt(ins[inputKey]?.tensorShape?.dim?.[1]?.size || expectedD, 10);
      }
      if (outKeys.length) outputName = outs[outKeys[0]]?.name || outputName;
    } catch(e){}

    // warmup
    tf.tidy(() => {
      const z = tf.zeros([1, D], "float32");
      const dict = { [inputKey]: z };
      const out = outputName ? model.execute(dict, outputName) : model.execute(dict);
      const y = Array.isArray(out) ? out[0] : out;
      y.dataSync();
    });

    modelStatus = "読込完了";
    return {ok:true, info:`backend=${tf.getBackend()} inputKey=${inputKey} D=${expectedD}`};
  } catch(e) {
    model=null;
    modelStatus="読込失敗";
    return {ok:false, error:(e?.message || String(e))};
  }
}

/* ------------------ message loop ------------------ */
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

    const start = Date.now();
    const deadline = start + timeMs;

    // UI pos -> internal
    const st0 = convertPos(msg.pos);

    let bestMove = null;
    let bestScore = null;
    let depthDone = 0;
    let pvMove = null;

    try {
      for (let d=1; d<=depthMax; d++){
        timeOrCancel(deadline, requestId);

        const r = chooseBestRoot(st0, d, useNN, deadline, requestId, pvMove);
        if (!r || !r.move) break;

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
          bestUci: `${bestMove.fromSq}->${bestMove.toSq}${bestMove.promote?"+":""}${bestMove.dropAbsId?("*"+bestMove.dropAbsId):""}`
        });
      }
    } catch(e) {
      // __TIME__ / __CANCEL__ は想定内
    }

    self.postMessage({
      type:"result",
      requestId,
      ok: !!bestMove,
      bestMove: bestMove ? toUiMove(bestMove) : null,
      bestScore,
      depthDone,
      timeMs: Date.now() - start,
      modelStatus
    });
  }
};
