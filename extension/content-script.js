// Job Copilot — content script (DOM inspector + paste-and-next)
// Scans form fields, sends them for analysis, and handles hotkey paste actions.
// NEVER auto-fills. NEVER reads non-active field values.

(() => {
  "use strict";

  let analysisResult = null;
  let activeFieldIndex = -1;
  let fieldElements = [];

  // ── SHA-256 hash via Web Crypto ───────────────────────────────

  async function sha256hex(text) {
    const encoder = new TextEncoder();
    const data = encoder.encode(text);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  // ── Field scanning ────────────────────────────────────────────

  function scanFormFields() {
    const fields = [];
    const elements = document.querySelectorAll(
      "input[name], input[id], textarea[name], textarea[id], select[name], select[id]"
    );

    for (const el of elements) {
      const name = el.name || el.id || "";
      if (!name) continue;

      const fieldId = el.id || el.name || `field_${fields.length}`;
      const label = findLabel(el);
      const inputType = el.tagName === "SELECT" ? "select" : el.type || "text";
      const autocomplete = el.autocomplete || null;
      const required = el.required || false;

      let options = [];
      if (el.tagName === "SELECT") {
        options = Array.from(el.options).map((opt) => ({
          value: opt.value,
          label: opt.textContent.trim(),
          selected: opt.selected,
        }));
      }

      // Compute current value hash (only for non-empty values).
      let currentValueHash = null;
      const val = el.value || "";
      if (val.trim()) {
        sha256hex(val).then((hash) => {
          el.dataset.jobCopilotHash = hash;
        });
      }

      fields.push({
        field_id: fieldId,
        label,
        input_type: inputType,
        selector: buildSelector(el),
        context_text: findContextText(el),
        required,
        current_value_hash: currentValueHash,
        autocomplete,
        options,
      });

      fieldElements.push(el);
    }

    return fields;
  }

  function findLabel(el) {
    // Try <label for="id">.
    if (el.id) {
      const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (label) return label.textContent.trim();
    }
    // Try parent <label>.
    const parentLabel = el.closest("label");
    if (parentLabel) return parentLabel.textContent.trim();
    // Try aria-label.
    if (el.getAttribute("aria-label")) return el.getAttribute("aria-label");
    // Try placeholder.
    if (el.placeholder) return el.placeholder;
    // Fall back to name/id.
    return el.name || el.id || "";
  }

  function findContextText(el) {
    // Look for sibling help text or preceding paragraph.
    const parent = el.closest(".field-group, .form-group, .form-field, [class*=field]");
    if (parent) {
      const help = parent.querySelector(
        ".help-text, .description, .hint, [class*=help], [class*=hint], small, .form-text"
      );
      if (help) return help.textContent.trim();
    }
    // Check preceding text node or element.
    const prev = el.previousElementSibling;
    if (prev && !["INPUT", "TEXTAREA", "SELECT"].includes(prev.tagName)) {
      return prev.textContent.trim().slice(0, 500);
    }
    return "";
  }

  function buildSelector(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;
    if (el.name) return `[name="${CSS.escape(el.name)}"]`;
    return "";
  }

  // ── Page context extraction ───────────────────────────────────

  function extractPageContext(maxLen = 2000) {
    // Get main text content, excluding scripts and styles.
    const clone = document.body.cloneNode(true);
    clone
      .querySelectorAll("script, style, noscript, iframe")
      .forEach((el) => el.remove());
    const text = clone.textContent.replace(/\s+/g, " ").trim();
    return text.slice(0, maxLen);
  }

  // ── Analysis submission ───────────────────────────────────────

  async function submitAnalysis() {
    const fields = scanFormFields();
    if (fields.length === 0) return;

    // Compute value hashes for fields with values.
    for (let i = 0; i < fields.length; i++) {
      const el = fieldElements[i];
      const val = el.value || "";
      if (val.trim()) {
        fields[i].current_value_hash = await sha256hex(val);
      }
    }

    const pageContext = extractPageContext();
    const requestId = crypto.randomUUID();

    chrome.runtime.sendMessage({
      kind: "analyze",
      payload: {
        url: location.href,
        company_hint: extractCompanyHint(),
        page_context: pageContext || null,
        fields,
        request_id: requestId,
      },
    });
  }

  function extractCompanyHint() {
    // Try to extract company name from page title or meta tags.
    const ogSite = document.querySelector('meta[property="og:site_name"]');
    if (ogSite) return ogSite.content;
    // Try title patterns like "Apply at Company" or "Company - Job Application".
    const title = document.title;
    const match = title.match(
      /(?:at|for|@)\s+(.+?)(?:\s*[-–—|]\s*(?:job|apply|career|application))/i
    );
    if (match) return match[1].trim();
    return null;
  }

  // ── Active field tracking ─────────────────────────────────────

  function highlightField(index) {
    // Remove previous highlight.
    if (activeFieldIndex >= 0 && activeFieldIndex < fieldElements.length) {
      const prev = fieldElements[activeFieldIndex];
      prev.style.outline = "";
      prev.removeAttribute("data-job-copilot-active");
    }
    if (index >= 0 && index < fieldElements.length) {
      const el = fieldElements[index];
      el.style.outline = "2px solid #ffae00";
      el.setAttribute("data-job-copilot-active", "true");
      el.focus();
      activeFieldIndex = index;
    }
  }

  function getPrefilledValue(index) {
    if (!analysisResult?.result?.prefilled) return null;
    const field = fieldElements[index];
    if (!field) return null;
    const fieldId = field.id || field.name;
    return analysisResult.result.prefilled.find((p) => p.field_id === fieldId);
  }

  // ── Paste via execCommand (the only mutation path) ────────────

  function pasteValue(el, value) {
    el.focus();
    // Select existing content.
    el.select?.();
    document.execCommand("insertText", false, value);
    // Dispatch events for frameworks to pick up.
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // ── Hotkey handlers ───────────────────────────────────────────

  function handlePasteAndNext() {
    if (activeFieldIndex < 0 || activeFieldIndex >= fieldElements.length) return;
    const prefilled = getPrefilledValue(activeFieldIndex);
    if (prefilled) {
      pasteValue(fieldElements[activeFieldIndex], prefilled.value);
    }
    // Advance to next field.
    if (activeFieldIndex + 1 < fieldElements.length) {
      highlightField(activeFieldIndex + 1);
    }
  }

  function handlePasteAndPrev() {
    if (activeFieldIndex < 0 || activeFieldIndex >= fieldElements.length) return;
    const prefilled = getPrefilledValue(activeFieldIndex);
    if (prefilled) {
      pasteValue(fieldElements[activeFieldIndex], prefilled.value);
    }
    // Step back to previous field.
    if (activeFieldIndex > 0) {
      highlightField(activeFieldIndex - 1);
    }
  }

  function handleSkipField() {
    if (activeFieldIndex < 0 || activeFieldIndex >= fieldElements.length) return;
    // Move to next field without pasting.
    if (activeFieldIndex + 1 < fieldElements.length) {
      highlightField(activeFieldIndex + 1);
    }
  }

  // ── Message listener ──────────────────────────────────────────

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.kind === "hotkey") {
      switch (msg.action) {
        case "paste-and-next":
          handlePasteAndNext();
          break;
        case "paste-and-prev":
          handlePasteAndPrev();
          break;
        case "skip-field":
          handleSkipField();
          break;
      }
      sendResponse({ ok: true });
      return false;
    }

    if (msg.kind === "analysisResult") {
      analysisResult = msg;
      // Start at the first field.
      highlightField(0);
      sendResponse({ ok: true });
      return false;
    }

    if (msg.kind === "apply") {
      // Side panel wants to apply a value to a specific field.
      const field = fieldElements.find(
        (el) => (el.id || el.name) === msg.field_id
      );
      if (field) {
        pasteValue(field, msg.value);
      }
      sendResponse({ ok: true });
      return false;
    }
  });

  // ── Initialize ────────────────────────────────────────────────

  // Wait for page to be fully loaded, then submit for analysis.
  if (document.readyState === "complete") {
    setTimeout(submitAnalysis, 500);
  } else {
    window.addEventListener("load", () => setTimeout(submitAnalysis, 500));
  }
})();
