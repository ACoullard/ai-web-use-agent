(maxSummaryChars) => {
  const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea']);
  const formTags = new Set(['input', 'select', 'textarea']);
  const interactiveRoles = new Set([
    'button', 'link', 'checkbox', 'radio', 'tab', 'menuitem', 'option', 'switch', 'textbox',
  ]);

  function isVisible(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
    return true;
  }

  function isInteractive(el) {
    const tag = el.tagName.toLowerCase();
    if (interactiveTags.has(tag)) return true;
    const role = el.getAttribute('role');
    if (role && interactiveRoles.has(role)) return true;
    if (el.hasAttribute('onclick')) return true;
    if (el.isContentEditable) return true;
    if (el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1') return true;
    return false;
  }

  function isSuppressed(el) {
    return el.closest('[aria-hidden="true"], [inert]') !== null;
  }

  // Hit-test the element's center against document.elementFromPoint. If the topmost
  // element at that point isn't this element (nor an ancestor/descendant of it),
  // something is painted over it - typically a modal, cookie banner, or toast - and a
  // click would be intercepted. Returns the covering element's tag, or null when
  // unoccluded or untestable (elementFromPoint only works for points inside the
  // viewport, so an element scrolled out of view can't be hit-tested and is treated
  // as unoccluded).
  function occludingTag(el, rect) {
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) return null;
    const top = document.elementFromPoint(cx, cy);
    // elementFromPoint ignores pointer-events:none nodes, so a purely decorative
    // overlay that doesn't actually block clicks won't be reported here.
    if (!top || top === el || el.contains(top) || top.contains(el)) return null;
    return top.tagName.toLowerCase();
  }

  // Best-effort detection of a modal/overlay that likely blocks the rest of the
  // page. Explicit dialog semantics are high-confidence; the geometry fallback
  // catches the common "custom div overlay + backdrop" pattern that carries no
  // ARIA at all. Returns a short human-readable description, or null.
  function detectModal() {
    const openDialog = document.querySelector('dialog[open]');
    if (openDialog && isVisible(openDialog)) {
      const name = accessibleName(openDialog);
      return name ? `an open <dialog> ("${name}")` : 'an open <dialog>';
    }
    const aria = Array.from(
      document.querySelectorAll('[aria-modal="true"], [role="dialog"], [role="alertdialog"]')
    ).find((el) => isVisible(el) && !isSuppressed(el));
    if (aria) {
      const name = accessibleName(aria);
      return name ? `a modal dialog ("${name}")` : 'a modal dialog';
    }
    // Geometry fallback: a positioned, high-z-index element covering most of the
    // viewport is almost always an overlay/backdrop rather than page content.
    const viewportArea = window.innerWidth * window.innerHeight;
    for (const el of all) {
      if (!isVisible(el)) continue;
      const style = window.getComputedStyle(el);
      if (style.position !== 'fixed' && style.position !== 'absolute') continue;
      const z = parseInt(style.zIndex, 10);
      if (!Number.isFinite(z) || z < 1) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width * rect.height < 0.5 * viewportArea) continue;
      return 'a full-screen overlay';
    }
    return null;
  }

  function accessibleName(el) {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    if (el.tagName.toLowerCase() === 'input' && el.placeholder) return el.placeholder.trim();
    const text = el.innerText || el.value || '';
    return text.trim().slice(0, 120);
  }

  function fieldIdentifier(el) {
    for (const attr of ['name', 'data-qa', 'data-testid', 'id', 'autocomplete']) {
      const raw = el.getAttribute(attr);
      if (raw && raw.trim()) return raw.trim().slice(0, 60);
    }
    return null;
  }

  // Drop the previous observation's handles - stale entries here would resolve to
  // detached nodes that are no longer on the page.
  window.__webagentElements = [];

  const elements = [];
  let index = 0;
  const all = document.querySelectorAll('*');
  for (const el of all) {
    if (!isInteractive(el)) continue;
    if (!isVisible(el)) continue;
    if (isSuppressed(el)) continue;
    index += 1;
    window.__webagentElements[index] = el;
    const tag = el.tagName.toLowerCase();
    const isFormControl = formTags.has(tag);
    // el.type normalizes a missing or unrecognized type attribute to 'text'.
    const inputType = tag === 'input' ? el.type : null;
    const isToggle = inputType === 'checkbox' || inputType === 'radio';
    const occludedBy = occludingTag(el, el.getBoundingClientRect());
    elements.push({
      index,
      tag,
      role: el.getAttribute('role') || null,
      input_type: inputType,
      name: accessibleName(el),
      value: 'value' in el ? (el.value || null) : null,
      field: isFormControl ? fieldIdentifier(el) : null,
      // el.href resolves to an absolute URL; the raw attribute may be relative.
      href: tag === 'a' && el.href ? el.href : null,
      options: tag === 'select'
        ? Array.from(el.options).map((o) => ({ value: o.value, label: o.text.trim() }))
        : null,
      checked: isToggle ? el.checked : null,
      required: isFormControl && !!el.required,
      disabled: ('disabled' in el && !!el.disabled) || el.getAttribute('aria-disabled') === 'true',
      occluded: occludedBy !== null,
      occluded_by: occludedBy,
    });
  }

  // Feed/canonical links live in <head>, not among interactive elements - e.g. RSS
  // autodiscovery (<link rel="alternate" type="application/rss+xml">) has no
  // visible clickable affordance at all.
  const headLinks = Array.from(document.querySelectorAll('link[rel="alternate"], link[rel="canonical"]'))
    .map((l) => ({
      rel: l.getAttribute('rel'),
      type: l.getAttribute('type') || null,
      href: l.href || null,
      title: l.getAttribute('title') || null,
    }));

  const fullBodyText = document.body ? document.body.innerText.trim() : '';
  const bodyText = fullBodyText.slice(0, maxSummaryChars);

  const modalDescription = detectModal();

  return {
    title: document.title,
    url: window.location.href,
    elements,
    head_links: headLinks,
    text_summary: bodyText,
    text_total_length: fullBodyText.length,
    modal_present: modalDescription !== null,
    modal_description: modalDescription,
  };
}
