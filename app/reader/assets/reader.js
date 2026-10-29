(function () {
  var d = document.documentElement;
  var themeBtn = document.getElementById("themeToggle");
  var sizeBtn = document.getElementById("sizeToggle");
  var sizeOrder = ["base", "l", "xl"];
  var sizeLabels = { base: "A", l: "A+", xl: "A++" };

  function getTheme() {
    return d.getAttribute("data-theme") || "default";
  }
  function setTheme(t) {
    d.setAttribute("data-theme", t);
    try {
      localStorage.setItem("readerTheme", t);
    } catch (e) {}
    updateTheme();
  }
  function getSize() {
    return d.getAttribute("data-size") || "base";
  }
  function setSize(s) {
    if (s === "base") d.removeAttribute("data-size");
    else d.setAttribute("data-size", s);
    try {
      localStorage.setItem("readerSize", s);
    } catch (e) {}
    updateSize();
  }
  function updateTheme() {
    if (!themeBtn) return;
    var t = getTheme();
    themeBtn.textContent = t === "default" ? "🌙" : "☀️";
    themeBtn.setAttribute("aria-pressed", String(t === "dark"));
  }
  function updateSize() {
    if (!sizeBtn) return;
    var s = getSize();
    if (sizeOrder.indexOf(s) === -1) s = "base";
    sizeBtn.textContent = sizeLabels[s] || "A";
  }
  try {
    var savedTheme = localStorage.getItem("readerTheme");
    if (savedTheme) d.setAttribute("data-theme", savedTheme);
    var savedSize = localStorage.getItem("readerSize");
    if (savedSize && savedSize !== "base") d.setAttribute("data-size", savedSize);
  } catch (e) {}
  if (themeBtn) {
    themeBtn.addEventListener("click", function () {
      setTheme(getTheme() === "dark" ? "default" : "dark");
    });
  }
  if (sizeBtn) {
    sizeBtn.addEventListener("click", function () {
      var idx = sizeOrder.indexOf(getSize());
      if (idx === -1) idx = 0;
      setSize(sizeOrder[(idx + 1) % sizeOrder.length]);
    });
  }
  updateTheme();
  updateSize();
  try {
    var params = new URLSearchParams(location.search);
    var p = params.get("p");
    if (p) {
      var el = document.getElementById(p);
      if (el) {
        el.classList.add("is-target");
        if (!location.hash) el.scrollIntoView();
      }
    }
  } catch (e) {}
})();
