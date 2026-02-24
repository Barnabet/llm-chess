const grid = document.getElementById("style-grid");
const STYLE_KEY = "llmChessPieceStyle";
const PIECES = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"];

const getStyle = () => localStorage.getItem(STYLE_KEY) || "cburnett";
const setStyle = (style) => localStorage.setItem(STYLE_KEY, style);

const formatName = (style) =>
  style
    .split(/[-_]/g)
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ""))
    .join(" ");

const renderCard = (style, activeStyle) => {
  const card = document.createElement("div");
  card.className = "style-card";
  if (style === activeStyle) {
    card.classList.add("active");
  }

  const title = document.createElement("h3");
  title.textContent = formatName(style);

  const preview = document.createElement("div");
  preview.className = "style-preview";
  PIECES.forEach((piece) => {
    const img = document.createElement("img");
    img.src = `/piece/${style}/${piece}.svg`;
    img.alt = `${style} ${piece}`;
    preview.appendChild(img);
  });

  const meta = document.createElement("div");
  meta.className = "style-meta";
  meta.innerHTML = `<span>${style}</span><span>${style === activeStyle ? "Selected" : ""}</span>`;

  card.appendChild(title);
  card.appendChild(preview);
  card.appendChild(meta);
  card.addEventListener("click", () => {
    setStyle(style);
    renderStyles(style);
  });
  return card;
};

const renderStyles = (activeStyle) => {
  grid.innerHTML = "";
  styles.forEach((style) => {
    grid.appendChild(renderCard(style, activeStyle));
  });
};

let styles = [];

const loadStyles = async () => {
  try {
    const response = await fetch("/api/piece-styles");
    const data = await response.json();
    styles = data.styles || [];
    const activeStyle = getStyle();
    renderStyles(activeStyle);
  } catch (error) {
    grid.innerHTML = "<div class=\"note\">Unable to load piece styles.</div>";
  }
};

loadStyles();
