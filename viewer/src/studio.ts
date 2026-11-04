/**
 * Viewer ↔ `worldgen serve` integration: a world selector when the API is
 * reachable, and a world picker when the default world cannot be loaded.
 * Everything here degrades to nothing on static hosting (no API).
 */
import { api, viewerHref } from './api';
import type { Health, WorldInfo } from './apiTypes';
import { h } from './dom';
import type { Child } from './dom';

export interface StudioInfo {
  health: Health;
  worlds: WorldInfo[];
}

let probe: Promise<StudioInfo | null> | null = null;

/** Probes `/api/health` (once per page) and lists the worlds; null when unavailable. */
export function studioInfo(): Promise<StudioInfo | null> {
  probe ??= (async () => {
    try {
      const health = await api.health(2500);
      if (!health || health.ok === false) return null;
      let worlds: WorldInfo[] = [];
      try {
        const res = await api.worlds();
        worlds = Array.isArray(res.worlds) ? res.worlds : [];
      } catch {
        // The selector still offers the default world.
      }
      return { health, worlds };
    } catch {
      return null;
    }
  })();
  return probe;
}

/** True when the page was opened with an explicit `?world=`. */
export function hasWorldParam(location: Location = window.location): boolean {
  return !!new URLSearchParams(location.search).get('world')?.trim();
}

function navigateToWorld(url: string | null): void {
  const u = new URL(window.location.href);
  if (url) u.searchParams.set('world', url);
  else u.searchParams.delete('world');
  window.location.assign(u.href);
}

/** "World [select]" row; switching navigates to `?world=<url>`. */
export function worldSwitcher(info: StudioInfo, base: URL): HTMLElement {
  const select = h('select', { class: 'select', title: 'Switch to another compiled world' });
  const here = base.href;
  let matched = false;
  const defaultWorld = info.health.default_world;
  if (defaultWorld) {
    const href = new URL(defaultWorld.endsWith('/') ? defaultWorld : `${defaultWorld}/`, window.location.href).href;
    const selected = href === here;
    matched ||= selected;
    select.append(h('option', { value: '', selected }, `default (${defaultWorld})`));
  }
  for (const w of info.worlds) {
    const href = new URL(w.url.endsWith('/') ? w.url : `${w.url}/`, window.location.href).href;
    const selected = href === here;
    matched ||= selected;
    select.append(h('option', { value: w.url, selected }, w.name));
  }
  if (!matched) {
    select.prepend(h('option', { value: '#', selected: true, disabled: true }, base.pathname || base.href));
  }
  select.addEventListener('change', () => {
    if (select.value === '#') return;
    navigateToWorld(select.value || null);
  });
  return h('label', { class: 'world-switch' }, h('span', null, 'World'), select);
}

/** Overlay content offering the available worlds (used when `world/` fails to load). */
export function worldPicker(info: StudioInfo, onDismiss: () => void): Child[] {
  const worlds = info.worlds;
  const list = h('ul', { class: 'picker-list' });
  for (const w of worlds) {
    const meta = [w.tile_count ? `${w.tile_count} tiles` : '', w.created_utc ? w.created_utc.slice(0, 10) : ''].filter(Boolean).join(' · ');
    list.append(h('li', null, h('a', { href: viewerHref(w.url) }, h('b', { class: 'mono' }, w.name), h('span', { class: 'muted small' }, meta))));
  }
  return [
    h('h2', null, worlds.length ? 'Pick a world' : 'No compiled worlds yet'),
    h(
      'p',
      { class: 'muted' },
      worlds.length
        ? 'No world is served at world/, but the WorldGen server has these:'
        : 'No world is served at world/ and the output root contains no compiled worlds. Generate one:',
    ),
    worlds.length ? list : null,
    h(
      'div',
      { class: 'btn-row' },
      h('a', { class: 'btn btn-primary', href: 'generate.html' }, '＋ New world'),
      h('button', { class: 'btn', onclick: onDismiss }, 'Dismiss'),
    ),
  ];
}
