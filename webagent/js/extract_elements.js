(() => {
  const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea']);
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

  function accessibleName(el) {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    if (el.tagName.toLowerCase() === 'input' && el.placeholder) return el.placeholder.trim();
    const text = el.innerText || el.value || '';
    return text.trim().slice(0, 120);
  }

  // Clear markers from any previous observation before reassigning.
  document.querySelectorAll('[data-webagent-index]').forEach((el) => el.removeAttribute('data-webagent-index'));

  const elements = [];
  let index = 0;
  const all = document.querySelectorAll('*');
  for (const el of all) {
    if (!isInteractive(el)) continue;
    if (!isVisible(el)) continue;
    index += 1;
    el.setAttribute('data-webagent-index', String(index));
    const tag = el.tagName.toLowerCase();
    elements.push({
      index,
      tag,
      role: el.getAttribute('role') || null,
      name: accessibleName(el),
      value: 'value' in el ? (el.value || null) : null,
      // el.href resolves to an absolute URL; the raw attribute may be relative.
      href: tag === 'a' && el.href ? el.href : null,
      options: tag === 'select'
        ? Array.from(el.options).map((o) => ({ value: o.value, label: o.text.trim() }))
        : null,
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

  const bodyText = document.body ? document.body.innerText.trim().slice(0, 2000) : '';

  return {
    title: document.title,
    url: window.location.href,
    elements,
    head_links: headLinks,
    text_summary: bodyText,
  };
})();
