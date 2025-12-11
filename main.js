// ==============================
//  将棋AI 試作 main.js
//  - 本将棋の初期局面を表示
//  - クリックで駒を動かせる（まだ合法手チェックなし）
//  - ここからAIロジックを足していく
// ==============================

// ---- 盤面・駒の表現 ----
// side: 'S' = 先手, 'G' = 後手
// type: 'FU','KY','KE','GI','KI','KA','HI','OU' など
function createPiece(side, type) {
  return { side, type };
}

// 本将棋の初期局面
function createInitialBoard() {
  const S = (type) => createPiece('S', type); // 先手
  const G = (type) => createPiece('G', type); // 後手

  // 9x9 の二次元配列 [y][x]
  // y=0 が上段（後手側）, y=8 が下段（先手側）
  return [
    // y = 0: 後手の後段
    [G('KY'), G('KE'), G('GI'), G('KI'), G('OU'), G('KI'), G('GI'), G('KE'), G('KY')],
    // y = 1: 後手の二段目（飛・角）
    [null, G('HI'), null, null, null, null, null, G('KA'), null],
    // y = 2: 後手の歩
    [G('FU'), G('FU'), G('FU'), G('FU'), G('FU'), G('FU'), G('FU'), G('FU'), G('FU')],
    // y = 3〜5: 空
    [null, null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null, null, null],
    // y = 6: 先手の歩
    [S('FU'), S('FU'), S('FU'), S('FU'), S('FU'), S('FU'), S('FU'), S('FU'), S('FU')],
    // y = 7: 先手の二段目（角・飛）
    [null, S('KA'), null, null, null, null, null, S('HI'), null],
    // y = 8: 先手の後段
    [S('KY'), S('KE'), S('GI'), S('KI'), S('OU'), S('KI'), S('GI'), S('KE'), S('KY')],
  ];
}

// ---- 文字表示用 ----
function pieceToChar(piece) {
  if (!piece) return '';
  const table = {
    FU: '歩',
    KY: '香',
    KE: '桂',
    GI: '銀',
    KI: '金',
    KA: '角',
    HI: '飛',
    OU: '王',
    TO: 'と',
    NY: '杏',
    NK: '圭',
    NG: '全',
    UM: '馬',
    RY: '龍',
  };
  return table[piece.type] || '?';
}

// ---- アプリの状態 ----
const state = {
  board: createInitialBoard(),
  sideToMove: 'S', // 'S' = 先手, 'G' = 後手
  selected: null, // { x, y } または null
};

// ---- DOM 取得 ----
const boardEl = document.getElementById('board');
const turnLabelEl = document.getElementById('turnLabel');
const messageEl = document.getElementById('message');
const resetButton = document.getElementById('resetButton');
const aiMoveButton = document.getElementById('aiMoveButton');

// ---- 盤面描画 ----
function render() {
  // 手番表示
  turnLabelEl.textContent = state.sideToMove === 'S' ? '先手' : '後手';

  // 盤面を再描画
  boardEl.innerHTML = '';

  for (let y = 0; y < 9; y++) {
    for (let x = 0; x < 9; x++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.x = String(x);
      cell.dataset.y = String(y);

      // 選択中マスのハイライト
      if (state.selected && state.selected.x === x && state.selected.y === y) {
        cell.classList.add('selected');
      }

      const piece = state.board[y][x];
      if (piece) {
        const span = document.createElement('span');
        span.className = 'piece ' + (piece.side === 'S' ? 'sente' : 'gote');
        span.textContent = pieceToChar(piece);
        cell.appendChild(span);
      }

      boardEl.appendChild(cell);
    }
  }
}

// ---- 盤面操作 ----
function isInsideBoard(x, y) {
  return x >= 0 && x < 9 && y >= 0 && y < 9;
}

function movePiece(from, to) {
  if (!isInsideBoard(from.x, from.y) || !isInsideBoard(to.x, to.y)) return;

  const piece = state.board[from.y][from.x];
  if (!piece) return;

  // 取りも含めて単純に上書き（合法手チェックは後で追加）
  state.board[to.y][to.x] = piece;
  state.board[from.y][from.x] = null;
}

// ---- クリック処理（人間の手） ----
function handleBoardClick(event) {
  const cell = event.target.closest('.cell');
  if (!cell) return;

  const x = Number(cell.dataset.x);
  const y = Number(cell.dataset.y);

  if (!isInsideBoard(x, y)) return;

  const clickedPiece = state.board[y][x];

  // 1. 何も選択していない状態
  if (!state.selected) {
    if (!clickedPiece) {
      setMessage('まず、自分の駒をクリックしてください。');
      return;
    }
    if (clickedPiece.side !== state.sideToMove) {
      setMessage(state.sideToMove === 'S'
        ? '今は先手番です。先手の駒を選んでください。'
        : '今は後手番です。後手の駒を選んでください。'
      );
      return;
    }

    // 自分の駒を選択
    state.selected = { x, y };
    setMessage('移動先のマスをクリックしてください。（まだルールチェックなし）');
    render();
    return;
  }

  // 2. すでに何か選んでいる状態
  const from = state.selected;

  // 同じマスをもう一度クリック → キャンセル
  if (from.x === x && from.y === y) {
    state.selected = null;
    setMessage('選択をキャンセルしました。別の駒を選んでください。');
    render();
    return;
  }

  // TODO: ここに「合法手かどうか」のチェックを今後追加する
  movePiece(from, { x, y });
  state.selected = null;
  state.sideToMove = state.sideToMove === 'S' ? 'G' : 'S';
  setMessage('仮のルールで移動しました。（まだ本当の将棋の動きはチェックしていません）');
  render();
}

// ---- メッセージ表示 ----
function setMessage(text) {
  if (messageEl) {
    messageEl.textContent = text;
  }
}

// ---- イベント登録 ----
// ★ここが前回抜けていたポイントです！★
boardEl.addEventListener('click', handleBoardClick);

resetButton.addEventListener('click', () => {
  state.board = createInitialBoard();
  state.sideToMove = 'S';
  state.selected = null;
  setMessage('初期局面に戻しました。自分の駒をクリックしてから移動先を選んでください。');
  render();
});

// 今後ここに「AIの思考」をつなぐ
aiMoveButton.addEventListener('click', () => {
  alert('ここに「AIの手を選ぶロジック（探索）」をこれから実装していきます。');
});

// ---- 初期描画 ----
render();
setMessage('自分の駒をクリックしてから、移動先のマスをクリックしてください。（まだルールチェックなし）');
