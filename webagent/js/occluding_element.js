// Hit-test an element's center and describe whatever is painted over it.
//
// Mirrors the hit-test occludingTag() runs in extract_elements.js, which annotates the
// snapshot; this one runs at action time and returns a fuller description (tag + id +
// classes) because it becomes the error message the model reads. Returns null when the
// element is clear, or when it can't be hit-tested - elementFromPoint only answers for
// points inside the viewport.
(el) => {
  const rect = el.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  if (cx < 0 || cy < 0 || cx > window.innerWidth || cy > window.innerHeight) return null;
  // elementFromPoint ignores pointer-events:none nodes, so a purely decorative overlay
  // that wouldn't actually swallow the click isn't reported here.
  const top = document.elementFromPoint(cx, cy);
  if (!top || top === el || el.contains(top) || top.contains(el)) return null;

  const id = top.id ? `#${top.id}` : '';
  const classes = typeof top.className === 'string' ? top.className.trim() : '';
  const cls = classes ? '.' + classes.split(/\s+/).slice(0, 3).join('.') : '';
  return `<${top.tagName.toLowerCase()}${id}${cls}>`;
}
