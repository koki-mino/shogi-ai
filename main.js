/* main.js v4 (UI thread)
   - 探索＋NN推論は worker.js に移動（UI固まらない）
*/

const MODEL_URL = "./shogi_eval_wars_tfjs/model.json";
const WORKER_URL = "./worker.js?v=20251217_v8";

let worker = null;
let workerReady = false;
let workerModelStatus = "未初期化";
let workerLastInfo = "";
let thinking = false;
let lastRequestId = 0;

/* ------------------ Shogi Model (same as before) ------------------ */
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
const elTimeMs = document.getElementById("timeMs");
const elUseNN = document.getElementById("useNN");

const btnReset = document.getElementById("btnReset");
const btnUndo = document.getElementById("btnUndo");
const btnAIMove = document.getElementById("btnAIMove");
const btnStop = document.getElementById("btnStop");

btnReset.onclick = () => resetGame();
btnUndo.onclick = () => undo();
btnAIMove.onclick = () => aiMove();
btnStop.onclick = () => stopAI();

function setStatus(extra = "") {
  const nn = elUseNN?.checked ? "ON" : "OFF";
  const legal = (board && hands && sideToMove) ? countLegalMoves(board, hands, sideToMove) : 0;
  const chk = (board && sideToMove && isKingInCheck(board, sideToMove)) ? "（王手）" : "";

  const w = workerReady ? "ready" : "not-ready";
  const busy = thinking ? "YES" : "NO";

  elStatus.textContent =
`手番: ${sideToMove || "-"} ${chk}
NN: ${nn}
Worker: ${w} / thinking: ${busy}
Worker model: ${workerModelStatus}
${workerLastInfo}
合法手数: ${legal}
${extra}`.trim();

  btnAIMove.disabled = !workerReady || thinking;
  btnStop.disabled = !thinking;
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
  if (thinking) return; // 思考中は操作を止めたいならON（外してもOK）

  // drop mode
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
  render();
  const legals = generateLegalMoves(board, hands, sideToMove);
  if (legals.length === 0) {
    const chk = isKingInCheck(board, sideToMove);
    alert(chk ? `詰み：${sideToMove}の負け` : "引き分け（合法手なし）");
  }
}

/* ------------------ Init / Undo ------------------ */
function resetGame() {
  board = makeEmptyBoard();
  hands = makeEmptyHands();
  sideToMove = 'S';
  selected = null;
  selectedDrop = null;
  history = [];
  setupInitialPosition(board);
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
  render();
}
resetGame();

/* ------------------ Worker ------------------ */
function initWorker() {
  worker = new Worker(WORKER_URL);
  workerReady = false;
  workerModelStatus = "初期化中...";
  workerLastInfo = "";
  setStatus();

  worker.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.type === "init_done") {
      workerReady = !!msg.ok;
      workerModelStatus = msg.ok ? "読込完了" : ("読込失敗: " + (msg.error || ""));
      workerLastInfo = msg.info ? `info: ${msg.info}` : "";
      setStatus();
      return;
    }

    if (msg.type === "progress") {
      // 任意：進捗を表示
      if (thinking) {
        workerLastInfo = `depth=${msg.depthDone}/${msg.depthMax} best=${msg.bestUci || "-"} score=${(msg.bestScore ?? "").toString()}`;
        setStatus("AI思考中...");
      }
      return;
    }

    if (msg.type === "result") {
      if (msg.requestId !== lastRequestId) return; // 古い結果は無視
      thinking = false;

      if (msg.ok && msg.bestMove) {
        pushHistory();
        ({board, hands, sideToMove} = applyMove(board, hands, sideToMove, msg.bestMove));
        selected = null;
        selectedDrop = null;
        afterMove();
        setStatus(`AI: depth_done=${msg.depthDone} time=${msg.timeMs}ms score=${msg.bestScore}`);
      } else {
        setStatus("AI: 結果なし / 中断");
      }
      return;
    }

    if (msg.type === "log") {
      // console.log("[worker]", msg.message);
      return;
    }
  };

  worker.onerror = (e) => {
    console.error("worker error:", e);
    workerReady = false;
    workerModelStatus = "Workerエラー";
    setStatus();
  };

  worker.postMessage({
    type: "init",
    modelUrl: MODEL_URL,
    backend: "wasm" // ここは 'cpu' でもOK
  });
}
initWorker();

function stopAI() {
  if (!worker || !thinking) return;
  worker.postMessage({ type: "cancel", requestId: lastRequestId });
  // UI側も止める（Workerは自前で止まる）
  thinking = false;
  setStatus("停止要求を送信しました");
}

function aiMove() {
  if (!workerReady || !worker) return;

  const depthMax = Math.max(1, Math.min(10, parseInt(elDepth.value || "4", 10)));
  const timeMs = Math.max(50, Math.min(5000, parseInt(elTimeMs.value || "500", 10)));
  const useNN = !!elUseNN.checked;

  thinking = true;
  lastRequestId++;
  setStatus("AI思考中...");

  const pos = {
    board: cloneBoard(board),
    hands: cloneHands(hands),
    sideToMove
  };

  worker.postMessage({
    type: "think",
    requestId: lastRequestId,
    pos,
    depthMax,
    timeMs,
    useNN
  });
}

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
  const G='G', S='S';
  b[0][0]={side:G,type:'KY'}; b[0][1]={side:G,type:'KE'}; b[0][2]={side:G,type:'GI'}; b[0][3]={side:G,type:'KI'};
  b[0][4]={side:G,type:'OU'}; b[0][5]={side:G,type:'KI'}; b[0][6]={side:G,type:'GI'}; b[0][7]={side:G,type:'KE'}; b[0][8]={side:G,type:'KY'};
  b[1][1]={side:G,type:'HI'}; b[1][7]={side:G,type:'KA'};
  for (let x=0;x<9;x++) b[2][x]={side:G,type:'FU'};

  b[8][0]={side:S,type:'KY'}; b[8][1]={side:S,type:'KE'}; b[8][2]={side:S,type:'GI'}; b[8][3]={side:S,type:'KI'};
  b[8][4]={side:S,type:'OU'}; b[8][5]={side:S,type:'KI'}; b[8][6]={side:S,type:'GI'}; b[8][7]={side:S,type:'KE'}; b[8][8]={side:S,type:'KY'};
  b[7][1]={side:S,type:'KA'}; b[7][7]={side:S,type:'HI'};
  for (let x=0;x<9;x++) b[6][x]={side:S,type:'FU'};
}

/* ------------------ Moves / Legal generation (same as before) ------------------ */
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

function inPromoZone(side, y){ return side==='S' ? (y<=2) : (y>=6); }
function mustPromote(pieceType, side, toY){
  if (pieceType === 'FU' || pieceType === 'KY') return side==='S' ? (toY===0) : (toY===8);
  if (pieceType === 'KE') return side==='S' ? (toY<=1) : (toY>=7);
  return false;
}
function canPromote(pieceType){ return !!PROM_MAP[pieceType]; }

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
