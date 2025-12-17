/* worker.js v6
   - make/unmake + TT(Zobrist) で高速化（深さが伸びやすい）
   - 打ち歩詰め判定を「逃げ手の有無だけ」高速チェックに置換（重さ対策）
   - TFJS GraphModel で NN 評価（葉）
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

function sqIndex(x, y) { return y*9 + x; }
function xyFromSq(sq){ return {x: sq % 9, y: (sq/9)|0}; }

let inputKey = "input_layer_5";
let outputName = "Identity:0";
let expectedD = 2283;

let cancelRequestId = -1;

/* ===================== Zobrist + TT ===================== */
const MASK64 = (1n<<64n) - 1n;

// TT size（大きいほど強くなるがメモリ増）: 2^19=524288
const TT_POW = 19;
const TT_SIZE = 1 << TT_POW;
const TT_MASK = (1n << BigInt(TT_POW)) - 1n;

// typed-array TT（ChromeならOK）
const TT_key = new BigUint64Array(TT_SIZE);     // hash
const TT_val = new Float64Array(TT_SIZE);       // value
const TT_dep = new Int16Array(TT_SIZE);         // depth
const TT_flg = new Int8Array(TT_SIZE);          // 0 EXACT, 1 LOWER, 2 UPPER
const TT_mov = new Int32Array(TT_SIZE);         // packed move (-1 = none)

const TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2;

// Zobrist tables
// pieceZ[(color*15 + absId)*81 + sq]
const pieceZ = new BigUint64Array(2 * 15 * 81);
// handZ[(color*7 + pi)*19 + count]  count 0..18
const handZ  = new BigUint64Array(2 * 7 * 19);
let sideZ = 0n;

let rngState = 0x123456789abcdef0n;
function rand64(){
  // LCG (64-bit)
  rngState = (rngState * 6364136223846793005n + 1442695040888963407n) & MASK64;
  return rngState;
}
function initZobrist(){
  for (let i=0;i<pieceZ.length;i++) pieceZ[i] = rand64();
  for (let i=0;i<handZ.length;i++)  handZ[i]  = rand64();
  sideZ = rand64();
}
initZobrist();

function pz(color, absId, sq){
  return pieceZ[((color*15 + absId) * 81) + sq];
}
function hz(color, handIndex, count){
  // count 0..18
  return handZ[((color*7 + handIndex) * 19) + count];
}

function packMove(mv){
  // bits:
  // to (0..127) 7bits
  // from (0..127) 7bits  (drop => 127)
  // promote 1bit
  // dropAbsId 4bits (0..14)
  // hasMove implicit by packed!=-1
  const to = mv.toSq & 127;
  const from = (mv.fromSq === -1 ? 127 : (mv.fromSq & 127));
  const prom = mv.promote ? 1 : 0;
  const drop = (mv.dropAbsId || 0) & 15;
  return (to) | (from<<7) | (prom<<14) | (drop<<15);
}
function unpackMove(packed){
  if (packed < 0) return null;
  const toSq = packed & 127;
  const from = (packed>>7) & 127;
  const promote = ((packed>>14)&1) === 1;
  const dropAbsId = (packed>>15) & 15;
  const fromSq = (from===127) ? -1 : from;
  return {fromSq, toSq, promote, dropAbsId: dropAbsId || 0};
}
function ttIndex(hash){
  return Number(hash & TT_MASK); // safe (mask < 2^19)
}
function ttProbe(hash, depth, alpha, beta){
  const i = ttIndex(hash);
  const k = TT_key[i];
  if (k !== hash) return {hit:false, best:null, alpha, beta};
  const d = TT_dep[i];
  const best = unpackMove(TT_mov[i]);

  if (d >= depth) {
    const v = TT_val[i];
    const f = TT_flg[i];
    if (f === TT_EXACT) return {hit:true, value:v, cut:true, best, alpha, beta};
    if (f === TT_LOWER) {
      if (v > alpha) alpha = v;
      if (alpha >= beta) return {hit:true, value:v, cut:true, best, alpha, beta};
    } else if (f === TT_UPPER) {
      if (v < beta) beta = v;
      if (alpha >= beta) return {hit:true, value:v, cut:true, best, alpha, beta};
    }
  }
  return {hit:true, cut:false, best, alpha, beta};
}
function ttStore(hash, depth, value, flag, bestMove){
  const i = ttIndex(hash);
  // 置換：深いもの優先（同深さなら上書き）
  if (TT_key[i] !== hash || TT_dep[i] <= depth) {
    TT_key[i] = hash;
    TT_dep[i] = depth;
    TT_val[i] = value;
    TT_flg[i] = flag;
    TT_mov[i] = bestMove ? packMove(bestMove) : -1;
  }
}

/* ===================== TFJS ===================== */
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

/* ===================== Internal ===================== */
function opp(side){ return side ^ 1; }

function typeNameFromAbsId(absId){ return ID_PTYPE[absId] || "FU"; }
function absIdFromTypeName(name){ return PTYPE_ID[name] || 1; }

function handIndexOfTypeName(t){
  return HAND_ORDER.indexOf(t); // -1 if not hand type
}
function unpromoteTypeName(tName){ return UNPROM_MAP[tName] || tName; }
function promoteTypeName(tName){ return PROM_MAP[tName] || tName; }

function inPromoZone(side, y){ return side===0 ? (y<=2) : (y>=6); }
function canPromote(tName){ return !!PROM_MAP[tName]; }
function mustPromote(tName, side, toY){
  if (tName === 'FU' || tName === 'KY') return side===0 ? (toY===0) : (toY===8);
  if (tName === 'KE') return side===0 ? (toY<=1) : (toY>=7);
  return false;
}

function isOwnPiece(pc, side){ return side===0 ? (pc>0) : (pc<0); }
function isFoePiece(pc, side){ return side===0 ? (pc<0) : (pc>0); }

function computeHash(board81, handsS, handsG, side){
  let h = 0n;
  for (let sq=0;sq<81;sq++){
    const pc = board81[sq];
    if (!pc) continue;
    const color = pc>0 ? 0 : 1;
    const absId = Math.abs(pc);
    h ^= pz(color, absId, sq);
  }
  for (let i=0;i<7;i++){
    const cS = handsS[i]||0, cG = handsG[i]||0;
    h ^= hz(0, i, cS);
    h ^= hz(1, i, cG);
  }
  if (side===1) h ^= sideZ;
  return h & MASK64;
}

/* ===================== Convert UI pos ===================== */
function convertPos(pos){
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

  const handsS = new Int16Array(7);
  const handsG = new Int16Array(7);
  for (let i=0;i<7;i++){
    const t = HAND_ORDER[i];
    handsS[i] = (pos.hands.S[t] || 0);
    handsG[i] = (pos.hands.G[t] || 0);
  }

  const side = (pos.sideToMove === 'S') ? 0 : 1;
  const hash = computeHash(board81, handsS, handsG, side);

  return { board81, handsS, handsG, side, kingSqS, kingSqG, hash };
}

/* ===================== Move gen ===================== */
function hasUnpromotedPawnOnFile(board81, side, fileX){
  const pawnId = PTYPE_ID.FU;
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

function genPseudoMoves(state){
  const {board81, side, handsS, handsG} = state;
  const moves = [];
  const forward = (side===0) ? -1 : 1;

  function pushMove(fromSq, toSq, promote){
    moves.push({fromSq, toSq, promote: !!promote});
  }
  function pushMoveWithPromo(fromSq, toSq, tName, fromY, toY){
    if (!canPromote(tName)) { pushMove(fromSq,toSq,false); return; }
    const promoPossible = inPromoZone(side, fromY) || inPromoZone(side, toY);
    const forced = mustPromote(tName, side, toY);
    if (!promoPossible) { pushMove(fromSq,toSq,false); return; }
    if (forced) { pushMove(fromSq,toSq,true); return; }
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

    const addStep = (dx,dy)=>{
      const nx=x+dx, ny=y+dy;
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
  for (let i=0;i<7;i++){
    const n = hands[i];
    if (n<=0) continue;
    const tName = HAND_ORDER[i];

    for (let y=0;y<9;y++) for (let x=0;x<9;x++){
      const toSq = sqIndex(x,y);
      if (board81[toSq]) continue;

      if (tName==='FU' || tName==='KY'){
        if ((side===0 && y===0) || (side===1 && y===8)) continue;
      }
      if (tName==='KE'){
        if ((side===0 && y<=1) || (side===1 && y>=7)) continue;
      }
      if (tName==='FU'){
        if (hasUnpromotedPawnOnFile(board81, side, x)) continue;
      }
      moves.push({fromSq:-1, toSq, dropAbsId: absIdFromTypeName(tName), promote:false});
    }
  }

  return moves;
}

/* ===================== Check detection ===================== */
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

/* ===================== make/unmake + hash update ===================== */
function makeMove(state, mv, stack){
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
    prevHash: state.hash,
    handIncType: 0,
  };

  // side-to-move toggle
  state.hash ^= sideZ;

  if (mv.fromSq === -1) {
    // drop
    const absId = mv.dropAbsId;
    const pc = (side===0) ? absId : -absId;

    // hand count change in hash
    const hands = (side===0) ? state.handsS : state.handsG;
    const tName = typeNameFromAbsId(absId);
    const hi = handIndexOfTypeName(tName);
    const oldC = hands[hi];
    const newC = oldC - 1;
    state.hash ^= hz(side, hi, oldC);
    state.hash ^= hz(side, hi, newC);
    hands[hi] = newC;

    // place piece hash
    state.hash ^= pz(side, absId, mv.toSq);
    board81[mv.toSq] = pc;
    rec.movedPc = pc;
  } else {
    const fromPc = board81[mv.fromSq];
    rec.movedPc = fromPc;

    const fromAbs = Math.abs(fromPc);
    const fromColor = side;

    // remove from-square piece hash
    state.hash ^= pz(fromColor, fromAbs, mv.fromSq);

    // capture
    if (rec.capPc) {
      const capColor = opp(side);
      const capAbs = Math.abs(rec.capPc);
      // remove captured piece hash
      state.hash ^= pz(capColor, capAbs, mv.toSq);

      const capName = typeNameFromAbsId(capAbs);
      const baseName = unpromoteTypeName(capName);
      const baseAbs = absIdFromTypeName(baseName);
      rec.handIncType = baseAbs;

      if (baseAbs !== PTYPE_ID.OU) {
        const hands = (side===0) ? state.handsS : state.handsG;
        const hi = handIndexOfTypeName(baseName);
        const oldC = hands[hi];
        const newC = oldC + 1;
        state.hash ^= hz(side, hi, oldC);
        state.hash ^= hz(side, hi, newC);
        hands[hi] = newC;
      }
    }

    board81[mv.fromSq] = 0;

    // moved piece (with promote)
    let tName = typeNameFromAbsId(fromAbs);
    if (mv.promote) tName = promoteTypeName(tName);
    const toAbs = absIdFromTypeName(tName);
    const toPc = (side===0) ? toAbs : -toAbs;

    // add to-square piece hash
    state.hash ^= pz(side, toAbs, mv.toSq);
    board81[mv.toSq] = toPc;

    if (toAbs === PTYPE_ID.OU) {
      if (side===0) state.kingSqS = mv.toSq;
      else state.kingSqG = mv.toSq;
    }
  }

  state.hash &= MASK64;
  state.side = opp(side);
  stack.push(rec);
  return rec;
}

function unmakeMove(state, stack){
  const rec = stack.pop();
  if (!rec) return;

  // restore everything cheaply
  state.side = rec.prevSide;
  state.kingSqS = rec.prevKingS;
  state.kingSqG = rec.prevKingG;
  state.hash = rec.prevHash;

  const {board81} = state;
  const side = rec.prevSide;

  if (rec.fromSq === -1) {
    board81[rec.toSq] = 0;
    // hands were restored by prevHash + state restore? (実体値も戻す必要あり)
    const absId = rec.dropAbsId;
    const tName = typeNameFromAbsId(absId);
    const hi = handIndexOfTypeName(tName);
    const hands = (side===0) ? state.handsS : state.handsG;
    hands[hi] += 1;
  } else {
    board81[rec.fromSq] = rec.movedPc;
    board81[rec.toSq] = rec.capPc;

    if (rec.handIncType && rec.handIncType !== PTYPE_ID.OU) {
      const baseName = typeNameFromAbsId(rec.handIncType);
      const hi = handIndexOfTypeName(baseName);
      const hands = (side===0) ? state.handsS : state.handsG;
      hands[hi] -= 1;
    }
  }
}

/* ===================== Fast uchi-fu-zume check ===================== */
function hasAnyEscapeFromCheck(state, sideInCheck, checkSq){
  // 逃げ手が1つでもあれば true
  const stack = [];

  const ksq = (sideInCheck===0) ? state.kingSqS : state.kingSqG;
  if (ksq < 0) return true;

  const {x:kx, y:ky} = xyFromSq(ksq);

  // (1) king moves / captures
  for (let dx=-1; dx<=1; dx++) for (let dy=-1; dy<=1; dy++){
    if (!dx && !dy) continue;
    const nx = kx+dx, ny = ky+dy;
    if (nx<0||nx>=9||ny<0||ny>=9) continue;
    const toSq = sqIndex(nx,ny);
    const tp = state.board81[toSq];
    if (tp && isOwnPiece(tp, sideInCheck)) continue;

    const mv = {fromSq: ksq, toSq, promote:false};
    makeMove(state, mv, stack);
    const ok = !isKingInCheck(state.board81, sideInCheck, state.kingSqS, state.kingSqG);
    unmakeMove(state, stack);
    if (ok) return true;
  }

  // (2) capture the checking pawn by any piece (including non-king)
  // 盤面を走査して「checkSqを攻撃できる駒」だけ試す
  const {x:tx, y:ty} = xyFromSq(checkSq);
  for (let sq=0;sq<81;sq++){
    const pc = state.board81[sq];
    if (!pc) continue;
    if (!isOwnPiece(pc, sideInCheck)) continue;

    // 擬似的に「この駒が checkSq を攻撃できるか」
    if (!attacksSquare(state.board81, sq, pc, tx, ty)) continue;

    const mv = {fromSq: sq, toSq: checkSq, promote:false};

    // 成りが絡む駒は「成る/成らない」両方試す（最低限）
    const absId = Math.abs(pc);
    const tName = typeNameFromAbsId(absId);
    const {y:fy} = xyFromSq(sq);

    const tries = [];
    if (canPromote(tName) && (inPromoZone(sideInCheck, fy) || inPromoZone(sideInCheck, ty)) && !mustPromote(tName, sideInCheck, ty)){
      tries.push({...mv, promote:false});
      tries.push({...mv, promote:true});
    } else if (canPromote(tName) && mustPromote(tName, sideInCheck, ty)){
      tries.push({...mv, promote:true});
    } else {
      tries.push(mv);
    }

    for (const m of tries){
      makeMove(state, m, stack);
      const ok = !isKingInCheck(state.board81, sideInCheck, state.kingSqS, state.kingSqG);
      unmakeMove(state, stack);
      if (ok) return true;
    }
  }

  return false;
}

/* ===================== Legal moves ===================== */
function genLegalMoves(state){
  const pseudo = genPseudoMoves(state);
  const legal = [];
  const stack = [];

  for (const mv of pseudo){
    makeMove(state, mv, stack);

    const mover = opp(state.side); // make後にsideが反転しているので、元の手番側
    const ok = !isKingInCheck(state.board81, mover, state.kingSqS, state.kingSqG);

    if (ok) {
      // 打ち歩詰め（軽量版）
      if (mv.fromSq === -1 && mv.dropAbsId === PTYPE_ID.FU) {
        const foe = state.side; // 今の手番（相手）
        if (isKingInCheck(state.board81, foe, state.kingSqS, state.kingSqG)) {
          // 「逃げ手が無い」なら禁止
          const hasEscape = hasAnyEscapeFromCheck(state, foe, mv.toSq);
          if (!hasEscape) {
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

/* ===================== eval ===================== */
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
  for (let i=0;i<7;i++){
    const t = HAND_ORDER[i];
    const v = VALUE[t] || 0;
    s += v * (state.handsS[i]||0);
    s -= v * (state.handsG[i]||0);
  }
  return s;
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

/* ===================== search ===================== */
function timeOrCancel(deadline, requestId){
  if (requestId === cancelRequestId) throw new Error("__CANCEL__");
  if (Date.now() >= deadline) throw new Error("__TIME__");
}

function orderMovesSimple(state, moves, ttMove){
  // TT move / capture / promote を軽量に優先（「毎手チェック判定」は重いのでやらない）
  const b = state.board81;
  const scored = moves.map(mv=>{
    let sc = 0;
    if (ttMove &&
        mv.fromSq===ttMove.fromSq && mv.toSq===ttMove.toSq &&
        (!!mv.promote)===(!!ttMove.promote) &&
        ((mv.dropAbsId||0)===(ttMove.dropAbsId||0))) sc += 100000;

    if (mv.fromSq !== -1) {
      const cap = b[mv.toSq];
      if (cap) {
        const capName = typeNameFromAbsId(Math.abs(cap));
        sc += (VALUE[capName]||0) + 200;
      }
      if (mv.promote) sc += 120;
    } else {
      sc += 15; // drop
      if (mv.dropAbsId === PTYPE_ID.FU) sc += 10;
    }
    return {mv, sc};
  });

  scored.sort((a,b)=>b.sc-a.sc);
  return scored.map(x=>x.mv);
}

function negamax(state, depth, alpha, beta, useNN, deadline, requestId){
  timeOrCancel(deadline, requestId);

  // TT probe
  const probe = ttProbe(state.hash, depth, alpha, beta);
  alpha = probe.alpha; beta = probe.beta;
  const ttMove = probe.best;
  if (probe.hit && probe.cut) return probe.value;

  const legals = genLegalMoves(state);
  if (legals.length===0){
    const inCheck = isKingInCheck(state.board81, state.side, state.kingSqS, state.kingSqG);
    return inCheck ? -999999 : 0;
  }
  if (depth<=0) return evalForSearch(state, useNN);

  const alpha0 = alpha;
  const ordered = orderMovesSimple(state, legals, ttMove);
  const stack=[];

  let best = -Infinity;
  let bestMove = ordered[0] || null;

  for (const mv of ordered){
    makeMove(state, mv, stack);
    const score = -negamax(state, depth-1, -beta, -alpha, useNN, deadline, requestId);
    unmakeMove(state, stack);

    if (score > best){
      best = score;
      bestMove = mv;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }

  let flag = TT_EXACT;
  if (best <= alpha0) flag = TT_UPPER;
  else if (best >= beta) flag = TT_LOWER;

  ttStore(state.hash, depth, best, flag, bestMove);
  return best;
}

function chooseBestRoot(state, depth, useNN, deadline, requestId){
  timeOrCancel(deadline, requestId);

  const probe = ttProbe(state.hash, depth, -Infinity, Infinity);
  const ttMove = probe.best;

  const legals = genLegalMoves(state);
  if (legals.length===0) return null;

  const ordered = orderMovesSimple(state, legals, ttMove);
  const stack=[];

  let bestMove = ordered[0];
  let bestScore = -Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (const mv of ordered){
    makeMove(state, mv, stack);
    const score = -negamax(state, depth-1, -beta, -alpha, useNN, deadline, requestId);
    unmakeMove(state, stack);

    if (score > bestScore){
      bestScore = score;
      bestMove = mv;
    }
    if (score > alpha) alpha = score;
  }

  ttStore(state.hash, depth, bestScore, TT_EXACT, bestMove);
  return {move: bestMove, score: bestScore};
}

function toUiMove(mv){
  if (mv.fromSq === -1){
    const to = xyFromSq(mv.toSq);
    return {from:null, to, drop: typeNameFromAbsId(mv.dropAbsId), promote:false};
  }
  const from = xyFromSq(mv.fromSq);
  const to = xyFromSq(mv.toSq);
  return {from, to, drop:null, promote: !!mv.promote};
}

/* ===================== init TF ===================== */
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

    tf.tidy(() => {
      const z = tf.zeros([1, D], "float32");
      const dict = { [inputKey]: z };
      const out = outputName ? model.execute(dict, outputName) : model.execute(dict);
      const y = Array.isArray(out) ? out[0] : out;
      y.dataSync();
    });

    modelStatus = "読込完了";
    return {ok:true, info:`backend=${tf.getBackend()} inputKey=${inputKey} D=${expectedD} TT=${TT_SIZE}`};
  } catch(e) {
    model=null;
    modelStatus="読込失敗";
    return {ok:false, error:(e?.message || String(e))};
  }
}

/* ===================== message loop ===================== */
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

    const st0 = convertPos(msg.pos);

    let bestMove = null;
    let bestScore = null;
    let depthDone = 0;

    try {
      for (let d=1; d<=depthMax; d++){
        timeOrCancel(deadline, requestId);

        const r = chooseBestRoot(st0, d, useNN, deadline, requestId);
        if (!r || !r.move) break;

        bestMove = r.move;
        bestScore = r.score;
        depthDone = d;

        self.postMessage({
          type:"progress",
          requestId,
          depthDone,
          depthMax,
          bestScore,
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
