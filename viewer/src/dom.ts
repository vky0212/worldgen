/**
 * Minimal DOM helpers shared by the viewer and the generate page. No three.js,
 * no Leaflet: importing this keeps a page bundle small.
 */

export type Child = Node | string | number | null | undefined | false;
export type Attrs = Record<string, unknown>;

/** Tiny hyperscript: h('div', {class: 'x', onclick: fn}, 'text', child). */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs | null = null,
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = String(v);
      else if (k === 'style') el.setAttribute('style', String(v));
      else if (k === 'dataset' && typeof v === 'object') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v as EventListener);
      else if (typeof v === 'boolean' || typeof v === 'number' || k === 'value') {
        (el as unknown as Record<string, unknown>)[k] = v;
      } else el.setAttribute(k, String(v));
    }
  }
  appendChildren(el, children);
  return el;
}

export function appendChildren(el: HTMLElement, children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el: HTMLElement): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

/** Collapsible sidebar section (a <details> element). Returns its body. */
export function createSection(parent: HTMLElement, title: string, open = true, extraClass = ''): HTMLDivElement {
  const body = h('div', { class: 'section-body' });
  const details = h('details', { class: `section ${extraClass}`.trim() }, h('summary', null, title), body);
  details.open = open;
  parent.append(details);
  return body;
}

export function checkbox(label: string, checked: boolean, onChange: (v: boolean) => void, extra?: Child): HTMLLabelElement {
  const input = h('input', { type: 'checkbox', checked });
  input.addEventListener('change', () => onChange(input.checked));
  return h('label', { class: 'check' }, input, h('span', null, label), extra ?? null);
}
