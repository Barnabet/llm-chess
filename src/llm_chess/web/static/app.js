const modelCountEl = document.getElementById("model-count");
const sessionCountEl = document.getElementById("session-count");
const activeGames = new Set();
const leaderboardSection = document.querySelector("[data-leaderboard]");
const leaderboardBody = document.querySelector("[data-leaderboard-body]");
const leaderboardMeta = document.querySelector("[data-leaderboard-meta]");
const sortButtons = Array.from(document.querySelectorAll(".sort-btn"));

const leaderboardState = {
  sortKey: "elo",
  sortDir: "desc",
  players: [],
};

const PIECE_MAP = {
  p: "bP",
  r: "bR",
  n: "bN",
  b: "bB",
  q: "bQ",
  k: "bK",
  P: "wP",
  R: "wR",
  N: "wN",
  B: "wB",
  Q: "wQ",
  K: "wK",
};

const BASE_COUNTS = {
  wP: 8,
  wN: 2,
  wB: 2,
  wR: 2,
  wQ: 1,
  wK: 1,
  bP: 8,
  bN: 2,
  bB: 2,
  bR: 2,
  bQ: 1,
  bK: 1,
};

const CAPTURE_ORDER = ["Q", "R", "B", "N", "P"];

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const STYLE_KEY = "llmChessPieceStyle";

const getPieceStyle = () => localStorage.getItem(STYLE_KEY) || "cburnett";

const updateSessionCount = () => {
  sessionCountEl.textContent = `${activeGames.size} active game${activeGames.size === 1 ? "" : "s"}`;
};

const formatElo = (value) => (typeof value === "number" ? value.toFixed(1) : "n/a");
const formatAccuracy = (value) => (typeof value === "number" ? `${value.toFixed(1)}%` : "n/a");
const formatCost = (value) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  if (value === 0) {
    return "$0.0000";
  }
  if (value < 0.0001) {
    return `$${value.toExponential(2)}`;
  }
  return `$${value.toFixed(6)}`;
};
const labelSide = (side) => {
  if (!side) {
    return "Player";
  }
  return `${side[0].toUpperCase()}${side.slice(1)}`;
};
const formatTime = (value) => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return `${value.toFixed(2)}s`;
};

const parseFen = (fen) => {
  const [placement] = fen.split(" ");
  const rows = placement.split("/");
  const pieces = {};
  rows.forEach((row, rowIndex) => {
    let fileIndex = 0;
    for (const char of row) {
      if (Number.isInteger(parseInt(char, 10))) {
        fileIndex += parseInt(char, 10);
      } else {
        const file = FILES[fileIndex];
        const rank = (8 - rowIndex).toString();
        pieces[`${file}${rank}`] = PIECE_MAP[char];
        fileIndex += 1;
      }
    }
  });
  return pieces;
};

class BoardView {
  constructor(container, onMove) {
    this.container = container;
    this.onMove = onMove;
    this.orientation = "white";
    this.interactive = false;
    this.turn = "white";
    this.humanColor = null;
    this.selected = null;
    this.pieces = {};
    this.lastMove = null;
    this.pieceStyle = getPieceStyle();
    this.squares = new Map();
    this.activeMover = null;
    this.build();
    this.updateOrientation();
  }

  setOrientation(orientation) {
    if (this.orientation !== orientation) {
      this.orientation = orientation;
      this.build();
      this.updateOrientation();
    }
  }

  setInteractive(interactive) {
    this.interactive = interactive;
  }

  setPieceStyle(style) {
    if (style && this.pieceStyle !== style) {
      this.pieceStyle = style;
      this.squares.forEach((square) => {
        square.dataset.piece = "";
      });
      this.renderPieces(this.pieces, this.lastMove ? this.lastMove.uci : null);
    }
  }

  setState({ fen, turn, humanColor, lastMove }) {
    const prevPieces = this.pieces;
    const prevLastMove = this.lastMove ? this.lastMove.uci : null;
    this.turn = turn;
    this.humanColor = humanColor;
    this.lastMove = lastMove;
    this.pieces = parseFen(fen);
    this.renderPieces(prevPieces, prevLastMove);
  }

  build() {
    this.container.innerHTML = "";
    this.squares.clear();
    const ranks = this.orientation === "white" ? [...RANKS].reverse() : [...RANKS];
    const files = this.orientation === "white" ? FILES : [...FILES].reverse();
    ranks.forEach((rank, rankIndex) => {
      files.forEach((file, fileIndex) => {
        const square = document.createElement("div");
        square.className = `square ${(rankIndex + fileIndex) % 2 === 0 ? "light" : "dark"}`;
        square.dataset.square = `${file}${rank}`;
        square.addEventListener("click", () => this.handleClick(square.dataset.square));
        this.container.appendChild(square);
        this.squares.set(square.dataset.square, square);
      });
    });
  }

  updateOrientation() {
    const frame = this.container.closest(".board-frame");
    if (frame) {
      frame.dataset.orientation = this.orientation;
    }
  }

  renderPieces(prevPieces = {}, prevLastMove = null) {
    const nextLastMove = this.lastMove ? this.lastMove.uci : null;
    let moving = null;
    if (nextLastMove && nextLastMove !== prevLastMove) {
      const from = nextLastMove.slice(0, 2);
      const to = nextLastMove.slice(2, 4);
      const movedPiece = prevPieces[from];
      if (movedPiece) {
        moving = { from, to, piece: movedPiece };
      }
    }

    let movingEl = null;
    if (moving && this.squares.has(moving.from)) {
      const fromSquare = this.squares.get(moving.from);
      movingEl = fromSquare.querySelector(".piece");
    }

    this.squares.forEach((square) => {
      square.classList.remove("selected", "last-move");
      const squareId = square.dataset.square;
      const nextPiece = this.pieces[squareId] || "";
      const domPieceEl = square.querySelector(".piece");
      const domPiece = domPieceEl ? domPieceEl.dataset.piece || "" : "";
      const currentPiece = square.dataset.piece || domPiece || "";
      if (square.dataset.piece !== currentPiece) {
        square.dataset.piece = currentPiece;
      }
      if (moving && (squareId === moving.from || squareId === moving.to)) {
        square.innerHTML = "";
        square.dataset.piece = "";
        return;
      }
      if (nextPiece === currentPiece) {
        return;
      }
      square.innerHTML = "";
      square.dataset.piece = nextPiece;
      if (nextPiece) {
        const piece = document.createElement("div");
        piece.className = "piece";
        piece.dataset.piece = nextPiece;
        piece.style.backgroundImage = `url(/piece/${this.pieceStyle}/${nextPiece}.svg)`;
        square.appendChild(piece);
      }
    });

    if (this.selected && this.squares.has(this.selected)) {
      this.squares.get(this.selected).classList.add("selected");
    }

    if (this.lastMove && this.lastMove.uci) {
      const from = this.lastMove.uci.slice(0, 2);
      const to = this.lastMove.uci.slice(2, 4);
      if (this.squares.has(from)) {
        this.squares.get(from).classList.add("last-move");
      }
      if (this.squares.has(to)) {
        this.squares.get(to).classList.add("last-move");
      }
    }

    if (moving) {
      this.animateMove(moving, movingEl, (floating) => {
        const target = this.squares.get(moving.to);
        if (!target) {
          return;
        }
        const nextPiece = this.pieces[moving.to];
        target.innerHTML = "";
        target.dataset.piece = nextPiece || "";
        if (!nextPiece) {
          if (floating) {
            floating.remove();
          }
          return;
        }
        const piece = floating || movingEl || document.createElement("div");
        piece.className = "piece";
        piece.dataset.piece = nextPiece;
        piece.style.backgroundImage = `url(/piece/${this.pieceStyle}/${nextPiece}.svg)`;
        piece.style.transform = "";
        piece.style.left = "";
        piece.style.top = "";
        piece.style.width = "";
        piece.style.height = "";
        piece.style.animation = "none";
        piece.classList.remove("moving-piece");
        target.appendChild(piece);
      });
    }
  }

  animateMove(moving, movingEl, onDone) {
    const fromSquare = this.squares.get(moving.from);
    const toSquare = this.squares.get(moving.to);
    if (!fromSquare || !toSquare) {
      onDone(null);
      return;
    }
    if (this.activeMover) {
      this.activeMover.remove();
      this.activeMover = null;
    }
    const boardRect = this.container.getBoundingClientRect();
    const fromRect = fromSquare.getBoundingClientRect();
    const toRect = toSquare.getBoundingClientRect();
    const squareSize = fromRect.width;
    const pieceSize = squareSize * 0.84;
    const offset = (squareSize - pieceSize) / 2;
    const floating = movingEl || document.createElement("div");
    floating.className = "piece moving-piece";
    floating.dataset.piece = moving.piece;
    floating.style.backgroundImage = `url(/piece/${this.pieceStyle}/${moving.piece}.svg)`;
    floating.style.width = `${pieceSize}px`;
    floating.style.height = `${pieceSize}px`;
    floating.style.left = `${fromRect.left - boardRect.left + offset}px`;
    floating.style.top = `${fromRect.top - boardRect.top + offset}px`;
    floating.style.transform = "translate(0px, 0px)";
    if (floating.parentElement !== this.container) {
      this.container.appendChild(floating);
    }
    this.activeMover = floating;
    const dx = toRect.left - fromRect.left;
    const dy = toRect.top - fromRect.top;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        floating.style.transform = `translate(${dx}px, ${dy}px)`;
      });
    });
    floating.addEventListener(
      "transitionend",
      () => {
        if (this.activeMover === floating) {
          this.activeMover = null;
        }
        onDone(floating);
      },
      { once: true }
    );
  }

  handleClick(square) {
    if (!this.interactive || !this.humanColor) {
      return;
    }
    if (this.turn !== this.humanColor) {
      return;
    }
    const piece = this.pieces[square];
    if (!this.selected) {
      if (!piece) {
        return;
      }
      const isWhite = piece.startsWith("w");
      if ((isWhite && this.humanColor !== "white") || (!isWhite && this.humanColor !== "black")) {
        return;
      }
      this.selected = square;
      this.renderPieces();
      return;
    }

    if (square === this.selected) {
      this.selected = null;
      this.renderPieces();
      return;
    }

    const from = this.selected;
    const to = square;
    this.selected = null;
    this.renderPieces();
    let uci = `${from}${to}`;
    const movingPiece = this.pieces[from];
    if (movingPiece && movingPiece.endsWith("P")) {
      const targetRank = to[1];
      if ((movingPiece.startsWith("w") && targetRank === "8") || (movingPiece.startsWith("b") && targetRank === "1")) {
        uci = `${uci}q`;
      }
    }
    this.onMove(uci);
  }
}

class Arena {
  constructor(root) {
    this.root = root;
    this.mode = root.dataset.arena;
    this.overlay = root.querySelector("[data-overlay]");
    this.board = new BoardView(root.querySelector("[data-board]"), (uci) => this.submitMove(uci));
    this.statusEl = root.querySelector("[data-status]");
    this.movesEl = root.querySelector("[data-moves]");
    this.banterEls = {
      white: root.querySelector('[data-player-banter="white"] .player-message__text'),
      black: root.querySelector('[data-player-banter="black"] .player-message__text'),
    };
    this.capturedEls = {
      white: root.querySelector('[data-captured="white"]'),
      black: root.querySelector('[data-captured="black"]'),
    };
    this.accuracyEl = root.querySelector("[data-accuracy]");
    this.errorEl = root.querySelector("[data-error]");
    this.evalFill = root.querySelector("[data-eval-fill]");
    this.evalText = root.querySelector("[data-eval-text]");
    this.noticeEl = root.querySelector("[data-board-notice]");
    this.noticeTitle = root.querySelector("[data-board-notice-title]");
    this.noticeText = root.querySelector("[data-board-notice-text]");
    this.noticeStatus = root.querySelector("[data-board-notice-status]");
    this.startBtn = root.querySelector("[data-start]");
    this.exitBtn = root.querySelector("[data-exit]");
    this.gameId = null;
    this.eventSource = null;
    this.state = null;
    this.playerCards = {
      white: {
        root: root.querySelector('[data-player-card="white"]'),
        name: root.querySelector('[data-player-card="white"] [data-player-name]'),
        elo: root.querySelector('[data-player-card="white"] [data-player-elo]'),
      },
      black: {
        root: root.querySelector('[data-player-card="black"]'),
        name: root.querySelector('[data-player-card="black"] [data-player-name]'),
        elo: root.querySelector('[data-player-card="black"] [data-player-elo]'),
      },
    };

    if (this.mode === "human") {
      this.modelSelect = root.querySelector("[data-model-select]");
      this.colorButtons = root.querySelectorAll(".toggle button");
      this.colorButtons.forEach((button) =>
        button.addEventListener("click", () => this.selectColor(button.dataset.color))
      );
    } else {
      this.whiteSelect = root.querySelector("[data-white-select]");
      this.blackSelect = root.querySelector("[data-black-select]");
      this.openingSelect = root.querySelector("[data-opening-select]");
    }

    this.startBtn.addEventListener("click", () => this.startGame());
    this.exitBtn.addEventListener("click", () => this.hideOverlay());
  }

  populateModels(models) {
    // Format model option with ELO label for Stockfish players
    const formatOption = (entry) => {
      const modelId = typeof entry === "string" ? entry : entry.model;
      if (entry.kind === "stockfish" && entry.starting_elo) {
        return `<option value="${modelId}">${modelId} (ELO ${entry.starting_elo})</option>`;
      }
      return `<option value="${modelId}">${modelId}</option>`;
    };

    if (this.mode === "human") {
      this.modelSelect.innerHTML = models.map(formatOption).join("");
    } else {
      const options = models.map(formatOption).join("");
      this.whiteSelect.innerHTML = options;
      this.blackSelect.innerHTML = options;
      if (models.length > 1) {
        this.blackSelect.selectedIndex = 1;
      }
    }
  }

  selectColor(color) {
    this.colorButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.color === color));
  }

  currentColor() {
    const active = this.root.querySelector(".toggle button.active");
    return active ? active.dataset.color : "white";
  }

  async startGame() {
    this.errorEl.textContent = "";
    this.movesEl.innerHTML = "";
    Object.values(this.banterEls).forEach((el) => {
      if (el) {
        el.textContent = "No heckles yet.";
      }
    });
    if (this.noticeEl) {
      this.noticeEl.setAttribute("hidden", "");
    }
    this.accuracyEl.textContent = "Accuracy n/a";
    const payload = { mode: this.mode === "human" ? "human_vs_llm" : "llm_vs_llm" };
    if (this.mode === "human") {
      payload.model = this.modelSelect.value;
      payload.humanColor = this.currentColor();
    } else {
      payload.whiteModel = this.whiteSelect.value;
      payload.blackModel = this.blackSelect.value;
      if (this.openingSelect && this.openingSelect.value !== "none") {
        payload.opening = this.openingSelect.value;
      }
    }

    try {
      const response = await fetch("/api/game", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        this.errorEl.textContent = data.error || "Unable to start game.";
        return;
      }
      this.gameId = data.id;
      activeGames.add(this.gameId);
      updateSessionCount();
      this.board.setInteractive(this.mode === "human");
      this.showOverlay();
      this.updateState(data);
      this.startStream();
    } catch (error) {
      this.errorEl.textContent = "Unable to reach the server.";
    }
  }

  startStream() {
    if (!this.gameId) {
      return;
    }
    if (this.eventSource) {
      this.eventSource.close();
    }
    this.eventSource = new EventSource(`/api/game/${this.gameId}/stream`);
    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.updateState(data);
        if (data.finished || data.error) {
          this.stopStream();
        }
      } catch (error) {
        this.errorEl.textContent = "Live updates paused.";
      }
    };
    this.eventSource.onerror = () => {
      this.errorEl.textContent = "Live updates paused.";
    };
  }

  stopStream() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  showOverlay() {
    this.overlay.classList.add("open");
  }

  hideOverlay() {
    this.overlay.classList.remove("open");
  }

  async submitMove(uci) {
    if (!this.gameId || this.mode !== "human") {
      return;
    }
    this.errorEl.textContent = "";
    try {
      const response = await fetch(`/api/game/${this.gameId}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uci }),
      });
      const data = await response.json();
      if (!response.ok) {
        this.errorEl.textContent = data.error || "Move rejected.";
        return;
      }
      this.updateState(data);
    } catch (error) {
      this.errorEl.textContent = "Unable to send your move.";
    }
  }

  updateState(state) {
    const previous = this.state;
    this.state = state;
    const humanColor = state.humanColor;
    const nextStyle = getPieceStyle();
    const styleBefore = this.board.pieceStyle;
    this.board.setPieceStyle(nextStyle);
    const previousMove = previous?.lastMove?.uci || null;
    const nextMove = state.lastMove?.uci || null;
    const boardChanged =
      !previous ||
      previous.fen !== state.fen ||
      previous.turn !== state.turn ||
      previous.humanColor !== humanColor ||
      previousMove !== nextMove;
    if (boardChanged) {
      this.board.setOrientation(humanColor || "white");
      this.board.setState({
        fen: state.fen,
        turn: state.turn,
        humanColor,
        lastMove: state.lastMove,
      });
    }
    if (boardChanged || nextStyle !== styleBefore) {
      this.renderCaptured(state.fen);
    }

    let status = `${state.turn === "white" ? "White" : "Black"} to move`;

    if (state.thinking) {
      status = "LLM thinking...";
    }

    if (state.error) {
      status = "Match halted";
      this.errorEl.textContent = state.error;
      activeGames.delete(state.id);
      updateSessionCount();
    }

    if (state.finished && !state.error) {
      status = `Game over | ${state.endReason || state.result || "draw"}`;
      activeGames.delete(state.id);
      updateSessionCount();
    }

    this.statusEl.textContent = status;

    const moveCount = state.moves ? state.moves.length : 0;
    const previousMoveCount = previous?.moves ? previous.moves.length : 0;
    if (moveCount !== previousMoveCount) {
      this.renderMoves(state.moves || []);
    }
    const banterBySide = this.extractBanterBySide(state.moves || []);
    const previousBanterBySide = this.extractBanterBySide(previous?.moves || []);
    ["white", "black"].forEach((side) => {
      const current = banterBySide[side];
      const previousEntry = previousBanterBySide[side];
      if (current?.text !== previousEntry?.text || current?.ply !== previousEntry?.ply) {
        this.renderBanter(side, current);
      }
    });
    this.renderAccuracy(state);
    this.renderEval(state);
    this.renderPlayerCards(state);
    this.renderToolNotice(state);
  }

  renderMoves(moves) {
    const shouldScroll = this.isMovesNearBottom();
    this.movesEl.innerHTML = "";
    moves.forEach((move) => {
      const row = document.createElement("div");
      const moveNumber = Math.ceil(move.ply / 2);
      const prefix = move.side === "white" ? `${moveNumber}.` : `${moveNumber}...`;
      const accuracyValue = typeof move.accuracy === "number" ? move.accuracy : null;
      const accuracy = accuracyValue !== null ? `${accuracyValue.toFixed(1)}%` : "n/a";
      row.className = `move-row ${move.side === "black" ? "move-row--black" : ""}`;
      const prefixEl = document.createElement("span");
      prefixEl.className = "move-row__ply";
      prefixEl.textContent = prefix;
      const sanEl = document.createElement("span");
      sanEl.className = "move-row__san";
      sanEl.textContent = move.san;
      const accEl = document.createElement("span");
      accEl.className = "move-row__acc";
      accEl.textContent = accuracy;
      if (accuracyValue !== null) {
        accEl.style.color = this.accuracyColor(accuracyValue);
      }
      row.append(prefixEl, sanEl, accEl);
      if (move.banter) {
        row.title = move.banter;
      }
      this.movesEl.appendChild(row);
    });
    if (shouldScroll) {
      this.scrollMovesToBottom();
    }
  }

  renderCaptured(fen) {
    if (!fen) {
      return;
    }
    const pieces = parseFen(fen);
    const counts = Object.fromEntries(Object.keys(BASE_COUNTS).map((key) => [key, 0]));
    Object.values(pieces).forEach((piece) => {
      if (counts[piece] !== undefined) {
        counts[piece] += 1;
      }
    });
    const style = getPieceStyle();
    const renderSide = (side) => {
      const container = this.capturedEls[side];
      if (!container) {
        return;
      }
      container.innerHTML = "";
      const colorPrefix = side === "white" ? "b" : "w";
      const captured = [];
      CAPTURE_ORDER.forEach((piece) => {
        const code = `${colorPrefix}${piece}`;
        const missing = Math.max(0, (BASE_COUNTS[code] || 0) - (counts[code] || 0));
        for (let i = 0; i < missing; i += 1) {
          captured.push(code);
        }
      });
      captured.forEach((code) => {
        const icon = document.createElement("div");
        icon.className = "captured-piece";
        icon.style.backgroundImage = `url(/piece/${style}/${code}.svg)`;
        container.appendChild(icon);
      });
    };
    renderSide("white");
    renderSide("black");
  }

  isMovesNearBottom() {
    if (!this.movesEl) {
      return false;
    }
    if (this.movesEl.scrollHeight <= this.movesEl.clientHeight) {
      return true;
    }
    const threshold = 80;
    return this.movesEl.scrollTop + this.movesEl.clientHeight >= this.movesEl.scrollHeight - threshold;
  }

  scrollMovesToBottom() {
    if (!this.movesEl) {
      return;
    }
    this.movesEl.scrollTop = this.movesEl.scrollHeight;
  }

  accuracyColor(value) {
    if (value > 100) {
      // Excellent move (improved position): purple/magenta (beyond blue)
      const excess = Math.min(value - 100, 100); // cap at 200% for color scaling
      const hue = 220 + (excess / 100) * 60; // 220 (blue) to 280 (purple)
      const saturation = 70 + (excess / 100) * 20; // 70-90%
      const lightness = 45 + (excess / 100) * 10; // 45-55%
      return `hsl(${hue} ${saturation}% ${lightness}%)`;
    }
    // 0-100%: red (0) → green (120) → blue (220)
    const clamped = Math.max(0, value);
    const hue = 220 * (clamped / 100);
    return `hsl(${hue} 70% 45%)`;
  }

  extractBanterBySide(moves) {
    const latest = { white: null, black: null };
    for (let i = moves.length - 1; i >= 0; i -= 1) {
      const entry = moves[i];
      if (!entry || typeof entry.banter !== "string") {
        continue;
      }
      const text = entry.banter.trim();
      if (!text) {
        continue;
      }
      if (entry.side === "white" && !latest.white) {
        latest.white = { ply: entry.ply, side: "white", text };
      }
      if (entry.side === "black" && !latest.black) {
        latest.black = { ply: entry.ply, side: "black", text };
      }
      if (latest.white && latest.black) {
        break;
      }
    }
    return latest;
  }

  renderBanter(side, entry) {
    const target = this.banterEls[side];
    if (!target) {
      return;
    }
    if (!entry) {
      target.textContent = "No heckles yet.";
      return;
    }
    const moveNumber = Math.ceil(entry.ply / 2);
    const prefix = entry.side === "white" ? `${moveNumber}.` : `${moveNumber}...`;
    target.textContent = `${prefix} ${entry.text}`;
  }

  renderAccuracy(state) {
    const summary = state.accuracySummary || {};
    const whiteAcc = summary[state.white?.model]?.avg;
    const blackAcc = summary[state.black?.model]?.avg;
    const format = (value) => (typeof value === "number" ? `${value.toFixed(1)}%` : "n/a");
    this.accuracyEl.textContent = `W ${format(whiteAcc)} | B ${format(blackAcc)}`;
  }

  renderToolNotice(state) {
    if (!this.noticeEl) {
      return;
    }
    const drawOffer = state.drawOffer;
    const lastTool = state.lastTool;
    let title = "";
    let text = "";
    let status = "";
    if (drawOffer) {
      const by = labelSide(drawOffer.by);
      title = `${by} proposed a draw`;
      text = drawOffer.message || "No explanation provided.";
      if (drawOffer.status === "accepted") {
        const responder = labelSide(drawOffer.response_by || "Opponent");
        status = `${responder} accepted the draw.`;
      } else if (drawOffer.status === "declined") {
        const responder = labelSide(drawOffer.response_by || "Opponent");
        status = `${responder} declined the draw.`;
      } else {
        status = "Awaiting opponent response.";
      }
    } else if (lastTool && lastTool.name === "resign") {
      const by = labelSide(lastTool.by);
      title = `${by} surrendered`;
      text = lastTool.reason || "";
      status = "";
    } else {
      this.noticeEl.setAttribute("hidden", "");
      return;
    }

    this.noticeTitle.textContent = title;
    this.noticeText.textContent = text;
    if (status) {
      this.noticeStatus.textContent = status;
      this.noticeStatus.removeAttribute("hidden");
    } else {
      this.noticeStatus.textContent = "";
      this.noticeStatus.setAttribute("hidden", "");
    }
    this.noticeEl.removeAttribute("hidden");
  }

  renderEval(state) {
    if (!this.evalFill || !this.evalText) {
      return;
    }
    const evaluation = state.evaluation || {};
    const type = evaluation.type;
    const rawValue = evaluation.value;
    let cp = typeof evaluation.cp === "number" ? evaluation.cp : null;
    let label = "Eval n/a";
    if (type === "mate" && typeof rawValue === "number") {
      label = `Mate ${rawValue > 0 ? "+" : ""}${rawValue}`;
      cp = rawValue > 0 ? 2000 : -2000;
    } else if (typeof cp === "number") {
      label = `${cp >= 0 ? "+" : ""}${(cp / 100).toFixed(2)}`;
    }
    const maxCp = 800;
    if (typeof cp !== "number") {
      this.evalFill.style.height = "50%";
      this.evalText.textContent = label;
      return;
    }
    const clamped = Math.max(-maxCp, Math.min(maxCp, cp));
    const whiteRatio = (clamped + maxCp) / (2 * maxCp);
    this.evalFill.style.height = `${whiteRatio * 100}%`;
    this.evalText.textContent = label;
  }

  renderPlayerCards(state) {
    const updateCard = (card, data, fallback) => {
      if (!card.root) {
        return;
      }
      const name = data?.name || data?.model || fallback;
      const elo = data?.elo;
      if (card.name) {
        card.name.textContent = name;
      }
      if (card.elo) {
        card.elo.textContent = `Elo ${formatElo(elo)}`;
      }
    };
    updateCard(this.playerCards.white, state.white || {}, "White");
    updateCard(this.playerCards.black, state.black || {}, "Black");
  }
}

const MODEL_COLORS = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
];

const modelLabel = (model) => {
  if (typeof model !== "string") {
    return "model";
  }
  const parts = model.split("/");
  return parts[parts.length - 1] || model;
};

const colorForModel = (model, map) => {
  if (map && map[model]) {
    return map[model];
  }
  return "#6f6a62";
};

const applySortState = () => {
  sortButtons.forEach((btn) => {
    const key = btn.dataset.sort;
    btn.classList.toggle("active", key === leaderboardState.sortKey);
    btn.classList.toggle("asc", key === leaderboardState.sortKey && leaderboardState.sortDir === "asc");
  });
};

const getSortValue = (entry, key) => {
  if (key === "elo") {
    return typeof entry.elo === "number" ? entry.elo : null;
  }
  if (key === "cost") {
    return typeof entry.cost === "number" ? entry.cost : null;
  }
  if (key === "accuracy") {
    return typeof entry.accuracy === "number" ? entry.accuracy : null;
  }
  if (key === "time") {
    return typeof entry.time === "number" ? entry.time : null;
  }
  return null;
};

const sortPlayers = (players) => {
  const { sortKey, sortDir } = leaderboardState;
  return [...players].sort((a, b) => {
    const aVal = getSortValue(a, sortKey);
    const bVal = getSortValue(b, sortKey);
    if (aVal === null && bVal === null) {
      return a.model.localeCompare(b.model);
    }
    if (aVal === null) {
      return 1;
    }
    if (bVal === null) {
      return -1;
    }
    if (aVal === bVal) {
      return a.model.localeCompare(b.model);
    }
    return sortDir === "asc" ? aVal - bVal : bVal - aVal;
  });
};

const formatAxisValue = (key, value) => {
  if (key === "cost") {
    return value < 0.001 ? value.toExponential(1) : value.toFixed(3);
  }
  if (key === "accuracy") {
    return value.toFixed(0);
  }
  if (key === "time") {
    return value.toFixed(2);
  }
  if (key === "elo") {
    return value.toFixed(0);
  }
  return value.toFixed(2);
};

const drawScatter = (svg, points, chart, colorMap) => {
  const width = 940;
  const height = 320;
  const margin = { top: 20, right: 24, bottom: 36, left: 44 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  if (!points.length) {
    const empty = document.createElementNS("http://www.w3.org/2000/svg", "text");
    empty.setAttribute("x", width / 2);
    empty.setAttribute("y", height / 2);
    empty.setAttribute("text-anchor", "middle");
    empty.setAttribute("fill", "#6f6a62");
    empty.textContent = "No data";
    svg.appendChild(empty);
    return;
  }

  const xVals = points.map((p) => p[chart.x]);
  const yVals = points.map((p) => p[chart.y]);
  let xMin = Math.min(...xVals);
  let xMax = Math.max(...xVals);
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const xPad = (xMax - xMin) * 0.1;
  const yPad = (yMax - yMin) * 0.1;
  xMin -= xPad;
  xMax += xPad;
  yMin -= yPad;
  yMax += yPad;

  const xScale = (value) =>
    margin.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const yScale = (value) =>
    margin.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;

  const axisGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  axisGroup.setAttribute("stroke", "rgba(0,0,0,0.1)");
  axisGroup.setAttribute("stroke-width", "1");
  svg.appendChild(axisGroup);

  for (let i = 0; i <= 4; i += 1) {
    const t = i / 4;
    const x = margin.left + t * plotWidth;
    const y = margin.top + t * plotHeight;
    const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    vLine.setAttribute("x1", x);
    vLine.setAttribute("y1", margin.top);
    vLine.setAttribute("x2", x);
    vLine.setAttribute("y2", margin.top + plotHeight);
    axisGroup.appendChild(vLine);
    const hLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    hLine.setAttribute("x1", margin.left);
    hLine.setAttribute("y1", y);
    hLine.setAttribute("x2", margin.left + plotWidth);
    hLine.setAttribute("y2", y);
    axisGroup.appendChild(hLine);
  }

  const labelGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
  labelGroup.setAttribute("fill", "#6f6a62");
  labelGroup.setAttribute("font-size", "10");
  svg.appendChild(labelGroup);

  for (let i = 0; i <= 4; i += 1) {
    const t = i / 4;
    const xValue = xMin + t * (xMax - xMin);
    const yValue = yMax - t * (yMax - yMin);
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("x", margin.left + t * plotWidth);
    xLabel.setAttribute("y", height - 12);
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.textContent = formatAxisValue(chart.x, xValue);
    labelGroup.appendChild(xLabel);
    const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yLabel.setAttribute("x", 10);
    yLabel.setAttribute("y", margin.top + t * plotHeight + 3);
    yLabel.textContent = formatAxisValue(chart.y, yValue);
    labelGroup.appendChild(yLabel);
  }

  const axisXLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  axisXLabel.setAttribute("x", margin.left + plotWidth / 2);
  axisXLabel.setAttribute("y", height - 2);
  axisXLabel.setAttribute("text-anchor", "middle");
  axisXLabel.setAttribute("fill", "#6f6a62");
  axisXLabel.setAttribute("font-size", "11");
  axisXLabel.textContent = chart.xLabel;
  svg.appendChild(axisXLabel);

  const axisYLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
  axisYLabel.setAttribute("x", 12);
  axisYLabel.setAttribute("y", margin.top - 6);
  axisYLabel.setAttribute("fill", "#6f6a62");
  axisYLabel.setAttribute("font-size", "11");
  axisYLabel.textContent = chart.yLabel;
  svg.appendChild(axisYLabel);

  points.forEach((point) => {
    const color = colorForModel(point.model, colorMap);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", xScale(point[chart.x]));
    circle.setAttribute("cy", yScale(point[chart.y]));
    circle.setAttribute("r", "5");
    circle.setAttribute("fill", color);
    circle.setAttribute("fill-opacity", "0.85");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${point.model}\nElo ${formatElo(point.elo)}\nCost ${formatCost(point.cost)}\nAccuracy ${formatAccuracy(point.accuracy)}\nTime ${formatTime(point.time)}`;
    circle.appendChild(title);
    svg.appendChild(circle);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", xScale(point[chart.x]) + 8);
    label.setAttribute("y", yScale(point[chart.y]) - 6);
    label.setAttribute("font-size", "10");
    label.setAttribute("fill", color);
    label.textContent = modelLabel(point.model);
    svg.appendChild(label);
  });
};

const renderCharts = (players) => {
  // Filter out stockfish players from charts (they don't have cost/time data)
  const chartPlayers = players.filter((entry) => !entry.model.startsWith("stockfish/"));

  const modelNames = Array.from(new Set(chartPlayers.map((entry) => entry.model))).sort();
  const colorMap = {};
  modelNames.forEach((model, index) => {
    colorMap[model] = MODEL_COLORS[index % MODEL_COLORS.length];
  });
  const charts = [
    { key: "cost-elo", x: "cost", y: "elo", xLabel: "Cost/Move ($)", yLabel: "Elo" },
    { key: "cost-accuracy", x: "cost", y: "accuracy", xLabel: "Cost/Move ($)", yLabel: "Accuracy %" },
    { key: "time-elo", x: "time", y: "elo", xLabel: "Avg Time/Move (s)", yLabel: "Elo" },
    { key: "accuracy-elo", x: "accuracy", y: "elo", xLabel: "Accuracy %", yLabel: "Elo" },
    { key: "time-accuracy", x: "time", y: "accuracy", xLabel: "Avg Time/Move (s)", yLabel: "Accuracy %" },
  ];
  charts.forEach((chart) => {
    const svg = document.querySelector(`[data-chart="${chart.key}"]`);
    if (!svg) {
      return;
    }
    const points = chartPlayers.filter(
      (entry) => Number.isFinite(entry[chart.x]) && Number.isFinite(entry[chart.y])
    );
    drawScatter(svg, points, chart, colorMap);
  });
};

const renderLeaderboard = () => {
  if (!leaderboardBody) {
    return;
  }
  const sorted = sortPlayers(leaderboardState.players);
  leaderboardBody.innerHTML = "";
  sorted.forEach((entry, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${index + 1}</td>
      <td><strong>${entry.model}</strong></td>
      <td>${formatElo(entry.elo)}</td>
      <td>${formatCost(entry.cost)}</td>
      <td>${formatTime(entry.time)}</td>
      <td>${formatAccuracy(entry.accuracy)}</td>
      <td>${entry.games ?? 0}</td>
      <td>${entry.wins ?? 0}/${entry.losses ?? 0}/${entry.draws ?? 0}</td>
    `;
    leaderboardBody.appendChild(row);
  });
  applySortState();
  renderCharts(sorted);
  if (leaderboardMeta) {
    leaderboardMeta.textContent = `${sorted.length} models tracked`;
  }
};

const loadLeaderboard = async () => {
  if (!leaderboardSection) {
    return;
  }
  try {
    const response = await fetch("/api/leaderboard");
    const data = await response.json();
    const players = data.players || {};
    leaderboardState.players = Object.entries(players).map(([model, stats]) => ({
      model,
      elo: typeof stats.elo === "number" ? stats.elo : null,
      cost: typeof stats.cost_avg === "number" ? stats.cost_avg : null,
      time: typeof stats.time_avg === "number" ? stats.time_avg : null,
      accuracy: typeof stats.accuracy_avg === "number" ? stats.accuracy_avg : null,
      games: stats.games ?? 0,
      wins: stats.wins ?? 0,
      losses: stats.losses ?? 0,
      draws: stats.draws ?? 0,
    }));
    renderLeaderboard();
  } catch (error) {
    if (leaderboardMeta) {
      leaderboardMeta.textContent = "Leaderboard unavailable";
    }
  }
};

sortButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.sort;
    if (!key) {
      return;
    }
    if (leaderboardState.sortKey === key) {
      leaderboardState.sortDir = leaderboardState.sortDir === "asc" ? "desc" : "asc";
    } else {
      leaderboardState.sortKey = key;
      leaderboardState.sortDir = "desc";
    }
    renderLeaderboard();
  });
});

const arenas = Array.from(document.querySelectorAll("[data-arena]")).map((root) => new Arena(root));

const loadModels = async () => {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    const models = data.models || [];
    modelCountEl.textContent = `${models.length} model${models.length === 1 ? "" : "s"}`;
    arenas.forEach((arena) => arena.populateModels(models));
  } catch (error) {
    modelCountEl.textContent = "Models unavailable";
  }
};

loadModels();
updateSessionCount();
loadLeaderboard();
