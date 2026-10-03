// App-wide keyboard behavior, independent of rendering. Dynamic text is never evaluated.
(() => {
  if (window.__rustInferKeyboard) return;
  window.__rustInferKeyboard = true;
  const focusable = root => [...root.querySelectorAll(
    'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]'
  )].filter(element => element.getClientRects().length && !element.closest('[inert]')
    && getComputedStyle(element).visibility !== 'hidden');
  const currentDialog = () => document.querySelector('.delete-confirm') || document.querySelector('.settings-dialog');
  const menuTrigger = menu => menu?.closest('.composer-attach-control')?.querySelector('button[aria-haspopup="menu"]');
  const menuItems = menu => [...menu.querySelectorAll('[role="menuitem"]:not(:disabled):not([aria-disabled="true"])')];
  const isolateDialog = dialog => {
    const changed = [];
    for (let branch = dialog; branch.parentElement && branch !== document.body; branch = branch.parentElement) {
      for (const sibling of branch.parentElement.children) {
        if (sibling !== branch && sibling instanceof HTMLElement && !sibling.inert) {
          sibling.inert = true;
          changed.push(sibling);
        }
      }
    }
    return () => changed.forEach(element => { element.inert = false; });
  };
  let dialogState;
  let openMenu;
  let openMenuTrigger;
  let menuEntryEdge = 'first';

  const observeFocus = () => {
    const dialog = currentDialog();
    if (dialogState?.element !== dialog) {
      if (dialogState) {
        const previous = dialogState.returnFocus;
        dialogState.restoreBackground();
        dialogState = null;
        if (previous?.isConnected) previous.focus();
        else if (!dialog) document.querySelector('.new-chat')?.focus();
      }
      if (dialog) {
        dialogState = { element: dialog, returnFocus: document.activeElement, restoreBackground: isolateDialog(dialog) };
        const initial = dialog.querySelector('#cancel-delete-conversation')
          || dialog.querySelector('#api-endpoint') || focusable(dialog)[0] || dialog;
        initial.focus();
      }
    }
    const menu = document.querySelector('.composer-menu');
    if (menu !== openMenu) {
      const oldMenu = openMenu;
      const oldTrigger = openMenuTrigger;
      openMenu = menu;
      openMenuTrigger = menuTrigger(menu);
      if (menu && !dialog) {
        const items = menuItems(menu);
        (menuEntryEdge === 'last' ? items.at(-1) : items[0])?.focus();
        if (!items.length) menu.focus();
        menuEntryEdge = 'first';
      } else if (oldMenu && !dialog && (oldMenu.contains(document.activeElement) || document.activeElement === document.body)) {
        oldTrigger?.focus();
      }
    }
  };
  const observer = new MutationObserver(observeFocus);
  observer.observe(document.documentElement, { childList: true, subtree: true });
  observeFocus();

  document.addEventListener('keydown', event => {
    const dialog = currentDialog();
    if (dialog) {
      if (event.key === 'Escape') {
        event.preventDefault();
        dialog.querySelector('#cancel-delete-conversation, #close-settings')?.click();
      } else if (event.key === 'Tab') {
        const targets = focusable(dialog);
        const first = targets[0], last = targets.at(-1);
        if (!targets.length) { event.preventDefault(); dialog.focus(); }
        else if (!dialog.contains(document.activeElement)) {
          event.preventDefault(); (event.shiftKey ? last : first).focus();
        } else if (event.shiftKey && document.activeElement === first) {
          event.preventDefault(); last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault(); first.focus();
        }
      }
      return;
    }

    const menu = document.querySelector('.composer-menu');
    const trigger = menuTrigger(menu);
    const active = document.activeElement;
    if (active?.matches('.composer-attach-control > button[aria-haspopup="menu"]')
      && ['ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault();
      menuEntryEdge = event.key === 'ArrowUp' ? 'last' : 'first';
      if (!menu) active.click();
      else {
        const items = menuItems(menu);
        (menuEntryEdge === 'last' ? items.at(-1) : items[0])?.focus();
      }
      return;
    }
    if (menu && event.key === 'Escape') {
      event.preventDefault();
      trigger?.click();
      trigger?.focus();
      return;
    }
    if (menu?.contains(active)) {
      const items = menuItems(menu);
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        if (!items.length) return;
        const index = items.indexOf(active);
        let next = index < 0 ? (event.key === 'ArrowUp' ? items.length - 1 : 0)
          : event.key === 'ArrowDown' ? (index + 1) % items.length
          : (index - 1 + items.length) % items.length;
        if (event.key === 'Home') next = 0;
        if (event.key === 'End') next = items.length - 1;
        items[next].focus();
        return;
      }
      if (event.key === 'Tab') {
        event.preventDefault();
        const targets = focusable(document).filter(element => !menu.contains(element));
        const next = event.shiftKey ? trigger : targets[targets.indexOf(trigger) + 1];
        trigger?.click();
        (next || trigger)?.focus();
        return;
      }
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'n') {
      event.preventDefault(); document.querySelector('.new-chat')?.click();
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); document.querySelector('.conversation-search input')?.focus();
    }
  });
})();
