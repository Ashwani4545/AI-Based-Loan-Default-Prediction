/* ═══════════════════════════════════════════════════════════════
<<<<<<< HEAD
   AegisBank — UI Script
=======
   Ministry of Banking — AegisBank
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
═══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  // ── FORM LOADING STATE ──────────────────────────────────────────────────
  const form = document.getElementById("loanForm");
<<<<<<< HEAD
  const btn  = document.getElementById("submitBtn");

  if (form && btn) {
    form.addEventListener("submit", (e) => {
      // Basic client-side validation before showing loader
=======
  const btn = document.getElementById("submitBtn");

  if (form && btn) {
    form.addEventListener("submit", (e) => {
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
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
<<<<<<< HEAD
        // Scroll to first error
=======
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
        const first = form.querySelector(".field-error");
        if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }

      btn.classList.add("loading");
      btn.disabled = true;
      btn.querySelector(".btn-label").textContent = "Analysing…";
    });
  }

<<<<<<< HEAD
  // ── NAV SCROLL SHADOW ───────────────────────────────────────────────────
  const nav = document.getElementById("mainNav");
  if (nav) {
    window.addEventListener("scroll", () => {
      nav.style.boxShadow = window.scrollY > 12
        ? "0 4px 32px rgba(0,0,0,0.45)"
        : "";
    }, { passive: true });
  }

=======
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
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

<<<<<<< HEAD
    const suffix   = raw.replace(/[0-9.]/g, "");
    const decimals = raw.includes(".") ? (raw.split(".")[1] || "").replace(/\D/g,"").length : 0;
    const dur = 1100;
    const t0  = performance.now();

    (function tick(now) {
      const p    = Math.min((now - t0) / dur, 1);
=======
    const suffix = raw.replace(/[0-9.]/g, "");
    const decimals = raw.includes(".") ? (raw.split(".")[1] || "").replace(/\D/g, "").length : 0;
    const dur = 1000;
    const t0 = performance.now();

    (function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = (num * ease).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  });

  // ── ANIMATE GAUGE ON RESULT PAGE ─────────────────────────────────────────
  const gaugeArc = document.getElementById("gaugeArc");
  if (gaugeArc) {
<<<<<<< HEAD
    const total  = 251.2;
=======
    const total = 251.2;
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
    const target = parseFloat(gaugeArc.getAttribute("stroke-dashoffset"));
    gaugeArc.setAttribute("stroke-dashoffset", total.toString());
    setTimeout(() => {
      gaugeArc.style.transition = "stroke-dashoffset 1.2s cubic-bezier(.16,1,.3,1)";
      gaugeArc.setAttribute("stroke-dashoffset", target.toString());
    }, 300);
  }

  // ── FIELD ERROR HIGHLIGHT ────────────────────────────────────────────────
  const style = document.createElement("style");
<<<<<<< HEAD
  style.textContent = `.field-error { border-color: #ef4444 !important; box-shadow: 0 0 0 3px rgba(239,68,68,.18) !important; }`;
  document.head.appendChild(style);

});
=======
  style.textContent = `.field-error { border-color: #dc2626 !important; box-shadow: 0 0 0 3px rgba(220,38,38,.12) !important; }`;
  document.head.appendChild(style);

});
>>>>>>> 5d6f7cb80e94c9b1113dea84a0f86173cb1c2f46
