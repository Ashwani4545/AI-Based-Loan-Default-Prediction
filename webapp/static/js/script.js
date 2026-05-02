/* ═══════════════════════════════════════════════════════════════
   Ministry of Banking — UI Script
═══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  // ── FORM LOADING STATE ──────────────────────────────────────────────────
  const form = document.getElementById("loanForm");
  const btn = document.getElementById("submitBtn");

  if (form && btn) {
    form.addEventListener("submit", (e) => {
      const required = form.querySelectorAll("[required]");
      let valid = true;
      required.forEach(el => {
        el.classList.remove("field-error");
        if (!el.value.trim()) {
          el.classList.add("field-error");
          valid = false;
        }
      });

      if (!valid) {
        e.preventDefault();
        const first = form.querySelector(".field-error");
        if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }

      btn.classList.add("loading");
      btn.disabled = true;
      btn.querySelector(".btn-label").textContent = "Analysing…";
    });
  }

  // ── RISK BAR ANIMATION (result page) ────────────────────────────────────
  const riskBar = document.querySelector(".rpt-risk-bar");
  if (riskBar) {
    const target = riskBar.style.width;
    riskBar.style.width = "0%";
    riskBar.style.transition = "none";
    requestAnimationFrame(() => {
      setTimeout(() => {
        riskBar.style.transition = "width 1.1s cubic-bezier(.16,1,.3,1)";
        riskBar.style.width = target;
      }, 200);
    });
  }

  // ── COUNTER ANIMATION (dashboard tiles) ─────────────────────────────────
  document.querySelectorAll(".mt-value").forEach(el => {
    const raw = el.textContent.trim();
    const num = parseFloat(raw.replace(/[^0-9.]/g, ""));
    if (isNaN(num) || num === 0) return;

    const suffix = raw.replace(/[0-9.]/g, "");
    const decimals = raw.includes(".") ? (raw.split(".")[1] || "").replace(/\D/g, "").length : 0;
    const dur = 1000;
    const t0 = performance.now();

    (function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = (num * ease).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  });

  // ── ANIMATE GAUGE ON RESULT PAGE ─────────────────────────────────────────
  const gaugeArc = document.getElementById("gaugeArc");
  if (gaugeArc) {
    const total = 251.2;
    const target = parseFloat(gaugeArc.getAttribute("stroke-dashoffset"));
    gaugeArc.setAttribute("stroke-dashoffset", total.toString());
    setTimeout(() => {
      gaugeArc.style.transition = "stroke-dashoffset 1.2s cubic-bezier(.16,1,.3,1)";
      gaugeArc.setAttribute("stroke-dashoffset", target.toString());
    }, 300);
  }

  // ── FIELD ERROR HIGHLIGHT ────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `.field-error { border-color: #dc2626 !important; box-shadow: 0 0 0 3px rgba(220,38,38,.12) !important; }`;
  document.head.appendChild(style);

});
