// Timestamp the page's last DOM change, so the controller can tell when the page has
// finished reacting to an action. Installed fresh per action; the previous watcher is
// disconnected first so at most one is ever attached.
() => {
  const previous = window.__webagentDomQuiet;
  if (previous) previous.observer.disconnect();

  const state = { lastMutation: performance.now() };
  state.observer = new MutationObserver(() => {
    state.lastMutation = performance.now();
  });
  state.observer.observe(document.documentElement, {
    subtree: true,
    childList: true,
    attributes: true,
    characterData: true,
  });
  window.__webagentDomQuiet = state;
}
