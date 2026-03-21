// AegisBank — Risk Engine UI Script

document.addEventListener("DOMContentLoaded", () => {

  // ── FORM SUBMIT LOADING STATE ──
  const form = document.getElementById("predictForm");
  const btn  = document.getElementById("submitBtn");

  if (form && btn) {
    form.addEventListener("submit", () => {
      btn.classList.add("loading");
      btn.disabled = true;
    });
  }

  // ── ANIMATE RISK FILL ON RESULT PAGE ──
  const riskFills = document.querySelectorAll(".risk-fill");
  riskFills.forEach(el => {
    const target = el.style.width;
    el.style.width = "0%";
    requestAnimationFrame(() => {
      setTimeout(() => { el.style.width = target; }, 150);
    });
  });

  // ── ANIMATE DONUT STROKE ──
  const donutCircle = document.querySelector(".donut-svg circle:last-child");
  if (donutCircle) {
    const target = donutCircle.getAttribute("stroke-dasharray");
    donutCircle.setAttribute("stroke-dasharray", "0 226");
    setTimeout(() => {
      donutCircle.setAttribute("stroke-dasharray", target);
    }, 300);
  }

  // ── STAT COUNTER ANIMATION ──
  document.querySelectorAll(".stat-value").forEach(el => {
    const text = el.textContent.trim();
    const num  = parseFloat(text.replace(/[^0-9.]/g, ""));
    if (isNaN(num)) return;

    const suffix = text.replace(/[0-9.]/g, "");
    const duration = 1200;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = (num * ease).toFixed(text.includes(".") ? 2 : 0) + suffix;
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });

  // ── LIVE INPUT DEBT-TO-INCOME HINT ──
  const incomeInput = document.querySelector('[name="income"]');
  const loanInput   = document.querySelector('[name="loan_amount"]');
  const emiInput    = document.querySelector('[name="existing_emi"]');

  function updateDTI() {
    if (!incomeInput || !loanInput) return;
    const income = parseFloat(incomeInput.value) || 0;
    const loan   = parseFloat(loanInput.value)   || 0;
    const emi    = parseFloat(emiInput?.value)    || 0;

    let hint = incomeInput.closest(".form-group")?.querySelector(".form-hint");
    if (!hint) return;

    if (income > 0 && loan > 0) {
      const monthly = income / 12;
      const estEMI  = loan / 36 + emi; // rough approximation
      const dti     = estEMI / monthly;
      const label   = dti < 0.4 ? "✅ Healthy" : dti < 0.6 ? "⚡ Moderate" : "⚠️ High";
      hint.textContent = `Estimated DTI: ${(dti * 100).toFixed(1)}% — ${label}`;
    } else {
      hint.textContent = "Gross annual income before taxes";
    }
  }

  [incomeInput, loanInput, emiInput].forEach(el => el?.addEventListener("input", updateDTI));

});