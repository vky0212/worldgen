// Unit tests for the pure routing module. Run with `npm test`
// (Node's built-in test runner + native TypeScript type stripping).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Router } from '../src/routing.ts';
import type { RouterEdge, RouterGraph, RouterRestriction } from '../src/routing.ts';

// Hand-made mini graph (local metres, y = north):
//
//            C(100,100) <==e4== E(200,100)      e4: one-way E -> C ("forward")
//    e6 .  /  |                  ^              e5: one-way D -> E (stored E->D, "backward")
//      .      | e2               | e5
//    A(0,0) --e1-- B(100,0) --e3-- D(200,0) ..e7.. F(300,0)
//
//  e6 (A-C) and e7 (D-F) are footpaths (no motor vehicles, no bicycles).
const nodes = [
  { id: 'A', x: 0, y: 0 },
  { id: 'B', x: 100, y: 0 },
  { id: 'C', x: 100, y: 100 },
  { id: 'D', x: 200, y: 0 },
  { id: 'E', x: 200, y: 100 },
  { id: 'F', x: 300, y: 0 },
];
const footOnly = { motor_vehicle: false, bicycle: false, foot: true };
const edges: RouterEdge[] = [
  { id: 'e1', from: 'A', to: 'B', direction: 'both', length_m: 100 },
  { id: 'e2', from: 'B', to: 'C', direction: 'both', length_m: 100 },
  { id: 'e3', from: 'B', to: 'D', direction: 'both', length_m: 100 },
  { id: 'e4', from: 'E', to: 'C', direction: 'forward', length_m: 100 },
  { id: 'e5', from: 'E', to: 'D', direction: 'backward', length_m: 100 },
  { id: 'e6', from: 'A', to: 'C', direction: 'both', length_m: 150, modes: footOnly },
  { id: 'e7', from: 'D', to: 'F', direction: 'both', length_m: 100, modes: footOnly },
];

// Heading east on e1 and turning north onto e2 at B is a left turn.
const noLeftAtB: RouterRestriction = {
  id: 'r:no_left',
  restriction: 'no_left_turn',
  kind: 'prohibitory',
  from_edge: 'e1',
  via_node: 'B',
  to_edge: 'e2',
  via_edges: [],
  enforced: true,
};

function graph(restrictions: RouterRestriction[] = []): RouterGraph {
  return { nodes, edges, restrictions };
}

const edgeIds = (r: { steps: { edgeId: string }[] }) => r.steps.map((s) => s.edgeId);

test('shortest path without restrictions', () => {
  const r = new Router(graph()).route('A', 'C', { mode: 'motor_vehicle' });
  assert.equal(r.found, true);
  assert.equal(r.length, 200);
  assert.deepEqual(edgeIds(r), ['e1', 'e2']);
  assert.deepEqual(r.nodeIds, ['A', 'B', 'C']);
});

test('forward one-way edge is only traversable from -> to', () => {
  const router = new Router(graph());
  const withFlow = router.route('E', 'C');
  assert.equal(withFlow.found, true);
  assert.deepEqual(edgeIds(withFlow), ['e4']);
  assert.equal(withFlow.length, 100);

  // Against the one-way: C -> B -> D -> E (e5 is usable D -> E).
  const against = router.route('C', 'E');
  assert.equal(against.found, true);
  assert.deepEqual(edgeIds(against), ['e2', 'e3', 'e5']);
  assert.equal(against.length, 300);
  assert.deepEqual(
    against.steps.map((s) => s.reversed),
    [true, false, true],
    'e2 traversed C->B (reversed), e5 traversed D->E (reversed w.r.t. stored E->D)',
  );
});

test('backward one-way edge is only traversable to -> from', () => {
  const router = new Router(graph());
  assert.deepEqual(edgeIds(router.route('D', 'E')), ['e5']);
  // E -> D must detour: E -> C (e4) -> B (e2) -> D (e3).
  const r = router.route('E', 'D');
  assert.deepEqual(edgeIds(r), ['e4', 'e2', 'e3']);
  assert.equal(r.length, 300);
});

test('ignoreOneway option allows travel against one-way edges', () => {
  const r = new Router(graph()).route('C', 'E', { ignoreOneway: true });
  assert.deepEqual(edgeIds(r), ['e4']);
});

test('pedestrians may walk against one-way traffic (matches the Python router)', () => {
  const r = new Router(graph()).route('C', 'E', { mode: 'foot' });
  assert.deepEqual(edgeIds(r), ['e4']);
});

test('enforced no_left_turn restriction forces a detour', () => {
  const router = new Router(graph([noLeftAtB]));
  assert.equal(router.stats.restrictionsEnforced, 1);
  const r = router.route('A', 'C', { mode: 'motor_vehicle' });
  assert.equal(r.found, true);
  // A -> B -> D -> E -> C (straight on at B, around the block).
  assert.deepEqual(edgeIds(r), ['e1', 'e3', 'e5', 'e4']);
  assert.equal(r.length, 400);
  // The restriction only concerns the e1 -> e2 manoeuvre: B -> C is still fine.
  assert.deepEqual(edgeIds(router.route('B', 'C')), ['e2']);
  // ... and can be ignored on request.
  assert.equal(router.route('A', 'C', { applyRestrictions: false }).length, 200);
});

test('restrictions with enforced=false are ignored', () => {
  const router = new Router(graph([{ ...noLeftAtB, enforced: false }]));
  assert.equal(router.stats.restrictionsNotEnforced, 1);
  assert.equal(router.route('A', 'C').length, 200);
});

test('mandatory restriction only allows the given continuation', () => {
  const onlyStraight: RouterRestriction = {
    id: 'r:only_straight',
    restriction: 'only_straight_on',
    kind: 'mandatory',
    from_edge: 'e1',
    via_node: 'B',
    to_edge: 'e3',
    enforced: true,
  };
  const router = new Router(graph([onlyStraight]));
  assert.deepEqual(edgeIds(router.route('A', 'C')), ['e1', 'e3', 'e5', 'e4']);
  assert.deepEqual(edgeIds(router.route('A', 'D')), ['e1', 'e3']);
});

test('via-way restrictions and dangling references are skipped, not applied', () => {
  const router = new Router(
    graph([
      { ...noLeftAtB, id: 'r:viaway', via_edges: ['e3'] },
      { ...noLeftAtB, id: 'r:dangling', to_edge: 'missing' },
    ]),
  );
  assert.equal(router.stats.restrictionsSkipped, 2);
  assert.equal(router.route('A', 'C').length, 200);
});

test('travel mode filters edges', () => {
  const router = new Router(graph([noLeftAtB]));
  const foot = router.route('A', 'C', { mode: 'foot' });
  assert.deepEqual(edgeIds(foot), ['e6']);
  assert.equal(foot.length, 150);

  const car = router.route('A', 'F', { mode: 'motor_vehicle' });
  assert.equal(car.found, false);
  assert.match(car.reason ?? '', /no motor vehicle route/);

  const walk = router.route('A', 'F', { mode: 'foot' });
  assert.equal(walk.found, true);
  assert.deepEqual(edgeIds(walk), ['e1', 'e3', 'e7']);
});

test('nearestNode snaps to the closest node usable by the mode', () => {
  const router = new Router(graph());
  assert.equal(router.nearestNode(95, 5), 'B');
  assert.equal(router.nearestNode(305, 0), 'F');
  // F is only reachable on foot, so a car snaps to D instead.
  assert.equal(router.nearestNode(305, 0, 'motor_vehicle'), 'D');
  assert.equal(router.nearestNode(305, 0, 'foot'), 'F');
});

test('no immediate U-turns except at dead ends', () => {
  // S =a=> M --c-- T, with M --b-- D. One-way a plus "no a->c at M" means the
  // only way from S to T is to drive to D, turn around and come back.
  const restriction: RouterRestriction = {
    id: 'r:no_a_c',
    kind: 'prohibitory',
    from_edge: 'a',
    via_node: 'M',
    to_edge: 'c',
    enforced: true,
  };
  const base: RouterGraph = {
    nodes: [
      { id: 'S', x: 0, y: 0 },
      { id: 'M', x: 10, y: 0 },
      { id: 'T', x: 20, y: 0 },
      { id: 'D', x: 10, y: 10 },
      { id: 'E', x: 10, y: 20 },
    ],
    edges: [
      { id: 'a', from: 'S', to: 'M', direction: 'forward', length_m: 10 },
      { id: 'b', from: 'M', to: 'D', direction: 'both', length_m: 10 },
      { id: 'c', from: 'M', to: 'T', direction: 'both', length_m: 10 },
    ],
    restrictions: [restriction],
  };
  // D is a dead end: turning around there is allowed.
  const deadEnd = new Router(base).route('S', 'T');
  assert.deepEqual(edgeIds(deadEnd), ['a', 'b', 'b', 'c']);
  assert.equal(deadEnd.length, 40);

  // Extend D with a spur to E: D is no longer a dead end, so the U-turn has to
  // happen at E instead (unless U-turns are explicitly allowed).
  const extended: RouterGraph = {
    ...base,
    edges: [...base.edges, { id: 'd', from: 'D', to: 'E', direction: 'both', length_m: 10 }],
  };
  const router = new Router(extended);
  assert.deepEqual(edgeIds(router.route('S', 'T')), ['a', 'b', 'd', 'd', 'b', 'c']);
  assert.deepEqual(edgeIds(router.route('S', 'T', { allowUTurns: true })), ['a', 'b', 'b', 'c']);
});

test('trivial and invalid requests', () => {
  const router = new Router(graph());
  const same = router.route('A', 'A');
  assert.equal(same.found, true);
  assert.equal(same.length, 0);
  assert.deepEqual(same.nodeIds, ['A']);
  const bad = router.route('A', 'nope');
  assert.equal(bad.found, false);
  assert.match(bad.reason ?? '', /not in the graph/);
});
