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

  function targetId() {
    var fromQuery = "";
    try {
      fromQuery = new URLSearchParams(location.search).get("p") || "";
    } catch (e) {}
    if (fromQuery) return fromQuery;
    if (location.hash && location.hash.length > 1) {
      try {
        return decodeURIComponent(location.hash.slice(1));
      } catch (e) {
        return location.hash.slice(1);
      }
    }
    return d.getAttribute("data-passage-id") || "";
  }

  function scrollToTarget() {
    var id = targetId();
    if (!id) return;
    var el = document.getElementById(id);
    if (!el) return;
    el.classList.add("is-target");
    el.scrollIntoView({ block: "start" });
    try {
      if (location.hash !== "#" + id) {
        history.replaceState(null, "", "#" + id);
      }
    } catch (e) {}
  }

  scrollToTarget();
  window.addEventListener("load", scrollToTarget);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(scrollToTarget);
  }
})();
