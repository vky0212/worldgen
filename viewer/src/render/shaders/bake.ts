/**
 * Procedural texture bake shaders. Every layer is a function of uv in [0,1)²
 * returning albedo, height, roughness, a tint mask (where the building colour
 * applies) and a glass mask (1 = vision glass; spandrels and signs are found by region). The bake
 * renders two arrays: albedo (rgb + glass) and normal/roughness/tint
 * (normal xy, roughness, tint). Facade tiles are one storey × one bay with
 * v = 0 at the floor; roof and ground tiles are seamless in both directions.
 *
 * Layer indices must match LAYER in ../buildingStyle.ts.
 */
import { NOISE_GLSL, PERIODIC_NOISE_GLSL } from './noise';

export const BAKE_VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const BAKE_FRAGMENT = /* glsl */ `
precision highp float;
uniform int uLayer;
uniform int uOutput;
uniform float uTexel;
varying vec2 vUv;

${NOISE_GLSL}
${PERIODIC_NOISE_GLSL}

struct Surf { vec3 alb; float h; float rough; float tint; float glass; };

Surf surfOf(vec3 alb, float h, float rough, float tint) { Surf s; s.alb = alb; s.h = h; s.rough = rough; s.tint = tint; s.glass = 0.0; return s; }

float inRect(vec2 p, vec2 a, vec2 b) { return step(a.x, p.x) * step(p.x, b.x) * step(a.y, p.y) * step(p.y, b.y); }

// Punched window with frame, optional mullions and a transom.
void windowAt(inout Surf s, vec2 uv, vec2 a, vec2 b, float frame, vec3 frameCol, float mullions, float transom) {
  if (inRect(uv, a, b) < 0.5) return;
  float d = min(min(uv.x - a.x, b.x - uv.x), min(uv.y - a.y, b.y - uv.y));
  bool isFrame = d < frame;
  float w = b.x - a.x;
  if (mullions > 0.0) {
    float m = fract((uv.x - a.x) / w * (mullions + 1.0));
    if (min(m, 1.0 - m) * w / (mullions + 1.0) < frame * 0.55) isFrame = true;
  }
  if (transom > 0.0 && abs(uv.y - transom) < frame * 0.55) isFrame = true;
  if (isFrame) { s.alb = frameCol; s.h = 0.6; s.rough = 0.45; s.tint = 0.0; s.glass = 0.0; }
  else { s.alb = vec3(0.035, 0.04, 0.045); s.h = 0.4; s.rough = 0.04; s.tint = 0.0; s.glass = 1.0; }
}

void sillAt(inout Surf s, vec2 uv, float x0, float x1, float y, vec3 col) {
  if (inRect(uv, vec2(x0 - 0.025, y - 0.035), vec2(x1 + 0.025, y)) > 0.5) { s.alb = col; s.h = 1.15; s.rough = 0.8; s.tint = 0.0; s.glass = 0.0; }
}

// Darker band along the floor line: dirt and the slab edge.
float grime(vec2 uv, float seed) {
  float streak = pNoise(vec2(uv.x * 23.0, 0.0), vec2(23.0, 1.0), seed);
  float below = smoothstep(0.32, 0.05, uv.y) * 0.08 * streak;
  return 1.0 - below - smoothstep(0.03, 0.0, uv.y) * 0.08;
}

// ------------------------------------------------------------ facades

Surf plaster(vec2 uv, bool windows) {
  float n = pFbm(uv, vec2(6.0), 1.0);
  float fine = pNoise(uv * 128.0, vec2(128.0), 2.0);
  Surf s = surfOf(vec3(0.86) * (0.88 + 0.16 * n) * (0.97 + 0.06 * fine) * grime(uv, 3.0), 0.95 + 0.05 * fine, 0.9, 1.0);
  if (uv.y < 0.025) { s.h = 1.05; s.alb *= 0.92; }
  if (windows) {
    windowAt(s, uv, vec2(0.29, 0.32), vec2(0.71, 0.8), 0.025, vec3(0.88, 0.88, 0.86), 1.0, -1.0);
    sillAt(s, uv, 0.29, 0.71, 0.32, vec3(0.78, 0.77, 0.74));
  }
  return s;
}

Surf ceramic(vec2 uv) {
  float row = floor(uv.y * 48.0);
  vec2 g = vec2(uv.x * 14.0 + mod(row, 2.0) * 0.5, uv.y * 48.0);
  vec2 f = fract(g);
  vec2 id = vec2(mod(floor(g.x), 14.0), row);
  float mortar = max(step(f.x, 0.07), step(f.y, 0.14));
  float v = 0.9 + 0.14 * prHash13(vec3(id, 4.0)) + 0.05 * pFbm(uv, vec2(4.0), 5.0);
  Surf s = surfOf(mix(vec3(0.85) * v, vec3(0.62, 0.61, 0.58), mortar) * grime(uv, 6.0), mortar > 0.5 ? 0.82 : 1.0, mortar > 0.5 ? 0.95 : 0.3, mortar > 0.5 ? 0.25 : 1.0);
  vec2 a = vec2(0.24, 0.3);
  vec2 b = vec2(0.76, 0.78);
  windowAt(s, uv, a, b, 0.018, vec3(0.74, 0.75, 0.77), 1.0, -1.0);
  // Iron security grille in front of the window.
  vec2 ga = a - 0.02;
  vec2 gb = b + 0.02;
  if (inRect(uv, ga, gb) > 0.5) {
    float bars = fract((uv.x - ga.x) / (gb.x - ga.x) * 11.0);
    float rails = abs(uv.y - (ga.y + gb.y) * 0.5);
    bool edge = min(min(uv.x - ga.x, gb.x - uv.x), min(uv.y - ga.y, gb.y - uv.y)) < 0.012;
    if (edge || min(bars, 1.0 - bars) < 0.06 || rails < 0.006) { s.alb = vec3(0.13, 0.13, 0.14); s.h = 1.3; s.rough = 0.55; s.tint = 0.0; s.glass = 0.0; }
  }
  return s;
}

Surf brick(vec2 uv) {
  float row = floor(uv.y * 44.0);
  vec2 g = vec2(uv.x * 12.0 + mod(row, 2.0) * 0.5, uv.y * 44.0);
  vec2 f = fract(g);
  vec2 id = vec2(mod(floor(g.x), 12.0), row);
  float mortar = max(step(f.x, 0.05), step(f.y, 0.16));
  float v = 0.82 + 0.3 * prHash13(vec3(id, 7.0));
  float n = pFbm(uv, vec2(8.0), 8.0);
  Surf s = surfOf(mortar > 0.5 ? vec3(0.74, 0.72, 0.67) : vec3(0.84) * v * (0.9 + 0.2 * n), mortar > 0.5 ? 0.8 : 1.0, 0.9, mortar > 0.5 ? 0.0 : 1.0);
  s.alb *= grime(uv, 9.0);
  windowAt(s, uv, vec2(0.3, 0.3), vec2(0.7, 0.8), 0.028, vec3(0.9, 0.9, 0.88), 1.0, 0.56);
  if (inRect(uv, vec2(0.27, 0.8), vec2(0.73, 0.86)) > 0.5) { s.alb = vec3(0.8, 0.78, 0.72); s.h = 1.08; s.rough = 0.8; s.tint = 0.0; s.glass = 0.0; }
  sillAt(s, uv, 0.3, 0.7, 0.3, vec3(0.8, 0.78, 0.72));
  return s;
}

Surf concretePanel(vec2 uv) {
  float n = pFbm(uv, vec2(5.0), 10.0);
  float streaks = pNoise(vec2(uv.x * 36.0, uv.y * 1.0), vec2(36.0, 1.0), 11.0);
  vec3 c = vec3(0.8) * (0.86 + 0.14 * n) * (1.0 - 0.1 * streaks * smoothstep(0.38, 0.0, uv.y));
  Surf s = surfOf(c, 0.96 + 0.04 * n, 0.88, 1.0);
  if (uv.x < 0.006 || uv.x > 0.994 || uv.y < 0.01) { s.alb *= 0.55; s.h = 0.85; }
  // Ribbon window: continuous band with mullions.
  if (uv.y > 0.38 && uv.y < 0.84) {
    float m = fract(uv.x * 2.0);
    bool frame = min(m, 1.0 - m) < 0.012 || uv.y < 0.395 || uv.y > 0.825;
    if (frame) { s.alb = vec3(0.3, 0.31, 0.32); s.h = 0.6; s.rough = 0.4; s.tint = 0.0; s.glass = 0.0; }
    else { s.alb = vec3(0.035, 0.04, 0.045); s.h = 0.45; s.rough = 0.04; s.tint = 0.0; s.glass = 1.0; }
  }
  return s;
}

Surf curtain(vec2 uv) {
  Surf s = surfOf(vec3(0.05, 0.055, 0.06), 0.5, 0.04, 1.0);
  s.glass = 1.0;
  if (uv.y > 0.03 && uv.y < 0.2) { s.glass = 0.0; s.alb = vec3(0.07, 0.075, 0.08); s.tint = 0.0; }
  bool mullion = uv.x < 0.025 || uv.x > 0.975 || uv.y < 0.025 || uv.y > 0.985;
  if (mullion) { s.alb = vec3(0.6, 0.62, 0.64); s.h = 0.9; s.rough = 0.35; s.tint = 0.0; s.glass = 0.0; }
  return s;
}

Surf metalSheet(vec2 uv) {
  float wave = 0.5 + 0.5 * sin(uv.x * 6.0 * 6.2831853);
  float n = pFbm(uv, vec2(4.0, 3.0), 12.0);
  float rust = smoothstep(0.62, 0.8, pFbm(uv + 0.37, vec2(3.0, 6.0), 13.0)) * smoothstep(0.6, 0.0, uv.y);
  vec3 c = vec3(0.84) * (0.92 + 0.08 * n);
  c = mix(c, vec3(0.45, 0.28, 0.18), rust * 0.5);
  Surf s = surfOf(c, 0.4 + 0.5 * wave, 0.45 + rust * 0.4, 1.0 - rust * 0.7);
  if (uv.y < 0.012) { s.alb *= 0.6; s.h = 0.3; }
  return s;
}

Surf timber(vec2 uv) {
  float f = fract(uv.y * 14.0);
  float board = floor(uv.y * 14.0);
  float grain = pFbm(vec2(uv.x, uv.y * 14.0 / 14.0), vec2(3.0, 56.0), 14.0 + board);
  vec3 c = vec3(0.82) * (0.75 + 0.35 * grain) * (0.92 + 0.12 * prHash12(vec2(board, 3.0)));
  Surf s = surfOf(c, 0.75 + 0.25 * f, 0.8, 1.0);
  if (f < 0.07) { s.alb *= 0.5; s.h = 0.7; }
  windowAt(s, uv, vec2(0.32, 0.33), vec2(0.68, 0.76), 0.03, vec3(0.22, 0.17, 0.13), 1.0, 0.56);
  return s;
}

Surf stone(vec2 uv) {
  float row = floor(uv.y * 5.0);
  float cells = 3.0 + floor(prHash12(vec2(row, 21.0)) * 2.0);
  float off = prHash12(vec2(row, 22.0));
  float gx = uv.x * cells + off;
  vec2 f = vec2(fract(gx), fract(uv.y * 5.0));
  vec2 id = vec2(mod(floor(gx), cells), row);
  float edge = min(min(f.x, 1.0 - f.x) * 3.0 / cells * 5.0, min(f.y, 1.0 - f.y));
  float mortar = step(edge, 0.03);
  float v = 0.85 + 0.25 * prHash13(vec3(id, 23.0));
  float n = pFbm(uv, vec2(10.0), 24.0);
  Surf s = surfOf(mortar > 0.5 ? vec3(0.62, 0.6, 0.56) : vec3(0.82) * v * (0.85 + 0.25 * n), mortar > 0.5 ? 0.6 : 0.85 + 0.15 * smoothstep(0.03, 0.12, edge) + 0.05 * n, 0.88, mortar > 0.5 ? 0.3 : 1.0);
  vec2 a = vec2(0.33, 0.3);
  vec2 b = vec2(0.67, 0.78);
  if (inRect(uv, a - 0.05, b + vec2(0.05, 0.07)) > 0.5) { s.alb = vec3(0.84, 0.82, 0.76) * (0.95 + 0.05 * n); s.h = 1.08; s.tint = 0.5; s.rough = 0.8; }
  windowAt(s, uv, a, b, 0.025, vec3(0.25, 0.2, 0.16), 1.0, 0.6);
  return s;
}

Surf shopfront(vec2 uv) {
  float n = pFbm(uv, vec2(5.0), 30.0);
  Surf s = surfOf(vec3(0.78) * (0.9 + 0.1 * n), 1.0, 0.85, 1.0);
  if (uv.y < 0.05) { s.alb = vec3(0.35, 0.35, 0.36); s.h = 1.05; s.tint = 0.0; }
  // Sign band (the building shader colours it per bay).
  if (uv.y > 0.82 && uv.y < 0.96 && uv.x > 0.03 && uv.x < 0.97) { s.alb = vec3(0.9); s.h = 1.15; s.rough = 0.5; s.tint = 0.0; s.glass = 0.0; }
  windowAt(s, uv, vec2(0.04, 0.05), vec2(0.96, 0.78), 0.012, vec3(0.16, 0.16, 0.17), 1.0, 0.68);
  return s;
}

// ------------------------------------------------------------ roofs (u across, v down the slope)

Surf roofClay(vec2 uv) {
  float course = fract(uv.y * 8.0);
  float cid = floor(uv.y * 8.0);
  float x = uv.x * 10.0 + mod(cid, 2.0) * 0.5;
  float barrel = 0.5 + 0.5 * cos(fract(x) * 6.2831853);
  float v = 0.85 + 0.25 * prHash13(vec3(mod(floor(x), 10.0), cid, 31.0));
  float n = pFbm(uv, vec2(4.0), 32.0);
  vec3 c = vec3(0.82) * v * (0.85 + 0.2 * n);
  float h = 0.35 + 0.45 * barrel + 0.2 * course;
  Surf s = surfOf(c * (0.7 + 0.3 * smoothstep(0.0, 0.15, course)), h, 0.65, 1.0);
  return s;
}

Surf roofSlate(vec2 uv) {
  float course = fract(uv.y * 12.0);
  float cid = floor(uv.y * 12.0);
  float x = uv.x * 8.0 + mod(cid, 2.0) * 0.5;
  float fx = fract(x);
  float v = 0.85 + 0.25 * prHash13(vec3(mod(floor(x), 8.0), cid, 33.0));
  bool gap = fx < 0.03 || fx > 0.97;
  Surf s = surfOf(vec3(0.75) * v * (gap ? 0.4 : 1.0), gap ? 0.3 : 0.4 + 0.6 * course, 0.55, 1.0);
  return s;
}

Surf roofMetal(vec2 uv) {
  float f = fract(uv.x * 5.0);
  float seam = smoothstep(0.03, 0.0, min(f, 1.0 - f));
  float n = pFbm(uv, vec2(3.0, 6.0), 34.0);
  return surfOf(vec3(0.82) * (0.93 + 0.07 * n) * (1.0 - 0.2 * seam), 0.5 + 0.5 * seam, 0.38, 1.0);
}

Surf roofFlat(vec2 uv) {
  float n = pFbm(uv, vec2(8.0), 35.0);
  float grit = prHash12(floor(uv * 512.0));
  Surf s = surfOf(vec3(0.78) * (0.85 + 0.15 * n) * (0.94 + 0.08 * grit), 0.5 + 0.05 * grit, 0.92, 1.0);
  // Drain stains.
  float stain = smoothstep(0.25, 0.0, length(fract(uv * 2.0) - 0.5)) * 0.25;
  s.alb *= 1.0 - stain;
  // Rooftop equipment: AC units and a water tank.
  vec2 p = uv * 8.0;
  if (inRect(p, vec2(1.0, 5.0), vec2(2.2, 5.8)) > 0.5 || inRect(p, vec2(5.5, 1.6), vec2(6.7, 2.4)) > 0.5) {
    s.alb = vec3(0.72, 0.73, 0.72); s.h = 1.0; s.rough = 0.5; s.tint = 0.0;
    vec2 q = fract(p * 3.0);
    if (length(q - 0.5) < 0.3) s.alb *= 0.45;
  }
  float tank = length(p - vec2(5.6, 5.8));
  if (tank < 0.75) { s.alb = vec3(0.82, 0.84, 0.86); s.h = 0.95 + 0.05 * (1.0 - tank / 0.75); s.rough = 0.4; s.tint = 0.0; }
  if (tank > 0.75 && tank < 0.95) s.alb *= 0.55;
  return s;
}

Surf roofShingle(vec2 uv) {
  float course = fract(uv.y * 16.0);
  float cid = floor(uv.y * 16.0);
  float x = uv.x * 6.0 + prHash12(vec2(cid, 36.0));
  float fx = fract(x);
  float v = 0.8 + 0.3 * prHash13(vec3(mod(floor(x), 6.0), cid, 37.0));
  float grit = prHash12(floor(uv * 400.0));
  bool slot = fx < 0.02 || fx > 0.98;
  return surfOf(vec3(0.7) * v * (0.9 + 0.2 * grit) * (slot ? 0.5 : 1.0) * (0.75 + 0.25 * smoothstep(0.0, 0.25, course)), 0.3 + 0.7 * course, 0.92, 1.0);
}

// ------------------------------------------------------------ ground

Surf grass(vec2 uv) {
  float n = pFbm(uv, vec2(4.0), 40.0);
  float m = pFbm(uv + 0.31, vec2(8.0), 41.0);
  float blades = pNoise(uv * vec2(256.0, 96.0), vec2(256.0, 96.0), 42.0);
  vec3 c = mix(vec3(0.16, 0.28, 0.07), vec3(0.32, 0.4, 0.12), n);
  c = mix(c, vec3(0.42, 0.4, 0.2), smoothstep(0.62, 0.8, m) * 0.6);
  c *= 0.8 + 0.35 * blades;
  return surfOf(c, 0.4 + 0.4 * blades + 0.2 * n, 0.95, 0.0);
}

Surf rock(vec2 uv) {
  float r = 0.0;
  float a = 0.5;
  vec2 per = vec2(3.0);
  for (int i = 0; i < 5; i++) {
    float n = pNoise(uv * per, per, 50.0 + float(i));
    r += a * (1.0 - abs(2.0 * n - 1.0));
    per *= 2.0;
    a *= 0.5;
  }
  float cracks = smoothstep(0.04, 0.0, pVoronoi(uv, vec2(6.0), 51.0).y - pVoronoi(uv, vec2(6.0), 51.0).x);
  float lichen = smoothstep(0.65, 0.8, pFbm(uv, vec2(6.0), 52.0));
  vec3 c = mix(vec3(0.36, 0.34, 0.31), vec3(0.55, 0.52, 0.48), r);
  c = mix(c, vec3(0.42, 0.45, 0.3), lichen * 0.5);
  c *= 1.0 - cracks * 0.45;
  return surfOf(c, r - cracks * 0.3, 0.85, 0.0);
}

Surf soil(vec2 uv) {
  float n = pFbm(uv, vec2(6.0), 60.0);
  vec3 v = pVoronoi(uv, vec2(24.0), 61.0);
  float pebble = smoothstep(0.32, 0.18, v.x) * step(0.7, v.z);
  vec3 c = mix(vec3(0.3, 0.22, 0.15), vec3(0.44, 0.34, 0.24), n);
  c = mix(c, vec3(0.42, 0.38, 0.32) * (0.85 + 0.3 * v.z), pebble * 0.6);
  return surfOf(c, 0.4 * n + 0.5 * pebble, 0.95, 0.0);
}

Surf asphalt(vec2 uv) {
  float g = prHash12(floor(uv * 512.0));
  float g2 = prHash12(floor(uv * 256.0) + 7.0);
  float n = pFbm(uv, vec2(4.0), 70.0);
  vec3 vor = pVoronoi(uv, vec2(5.0), 71.0);
  float crack = smoothstep(0.025, 0.0, vor.y - vor.x) * step(0.55, pFbm(uv, vec2(3.0), 72.0));
  vec3 c = vec3(0.115, 0.115, 0.12) * (0.8 + 0.4 * n);
  c += vec3(0.12) * step(0.93, g) - vec3(0.04) * step(g2, 0.1);
  c *= 1.0 - crack * 0.5;
  return surfOf(c, 0.5 + 0.3 * g - crack * 0.4, 0.82 + 0.1 * g, 0.0);
}

Surf gravel(vec2 uv) {
  vec3 v = pVoronoi(uv, vec2(40.0), 80.0);
  float dome = smoothstep(0.55, 0.0, v.x);
  vec3 c = mix(vec3(0.48, 0.45, 0.4), vec3(0.68, 0.64, 0.57), v.z) * (0.6 + 0.4 * dome);
  return surfOf(c, dome, 0.9, 0.0);
}

Surf forest(vec2 uv) {
  float n = pFbm(uv, vec2(6.0), 90.0);
  vec3 v = pVoronoi(uv, vec2(30.0), 91.0);
  float leaf = smoothstep(0.4, 0.2, v.x);
  vec3 c = mix(vec3(0.18, 0.13, 0.08), vec3(0.34, 0.25, 0.14), v.z);
  c = mix(c, vec3(0.16, 0.22, 0.1), smoothstep(0.55, 0.75, n) * 0.6);
  return surfOf(c * (0.7 + 0.3 * leaf), 0.3 * n + 0.4 * leaf, 0.95, 0.0);
}

Surf concretePlain(vec2 uv) {
  float n = pFbm(uv, vec2(5.0), 100.0);
  float g = prHash12(floor(uv * 384.0));
  vec3 c = vec3(0.6, 0.59, 0.57) * (0.85 + 0.15 * n) * (0.95 + 0.06 * g);
  float joint = step(fract(uv.x * 2.0), 0.006) + step(fract(uv.y * 2.0), 0.006);
  c *= 1.0 - 0.35 * min(joint, 1.0);
  return surfOf(c, 0.6 + 0.1 * g - 0.3 * min(joint, 1.0), 0.85, 0.0);
}

Surf surf(int layer, vec2 uv) {
  uv = fract(uv);
  if (layer == 0) return plaster(uv, true);
  if (layer == 1) return ceramic(uv);
  if (layer == 2) return brick(uv);
  if (layer == 3) return concretePanel(uv);
  if (layer == 4) return curtain(uv);
  if (layer == 5) return metalSheet(uv);
  if (layer == 6) return timber(uv);
  if (layer == 7) return stone(uv);
  if (layer == 8) return shopfront(uv);
  if (layer == 9) return plaster(uv, false);
  if (layer == 10) return roofClay(uv);
  if (layer == 11) return roofSlate(uv);
  if (layer == 12) return roofMetal(uv);
  if (layer == 13) return roofFlat(uv);
  if (layer == 14) return roofShingle(uv);
  if (layer == 15) return grass(uv);
  if (layer == 16) return rock(uv);
  if (layer == 17) return soil(uv);
  if (layer == 18) return asphalt(uv);
  if (layer == 19) return gravel(uv);
  if (layer == 20) return forest(uv);
  return concretePlain(uv);
}

void main() {
  vec2 uv = vUv;
  Surf s = surf(uLayer, uv);
  if (uOutput == 0) {
    gl_FragColor = vec4(clamp(s.alb, 0.0, 1.0), s.glass);
  } else {
    float t = uTexel;
    float hl = surf(uLayer, uv - vec2(t, 0.0)).h;
    float hr = surf(uLayer, uv + vec2(t, 0.0)).h;
    float hd = surf(uLayer, uv - vec2(0.0, t)).h;
    float hu = surf(uLayer, uv + vec2(0.0, t)).h;
    // Height units are ~2 cm; scale the slope by the texel size so the bevels keep their width in texels.
    float k = 0.35 / (t * 64.0);
    vec3 n = normalize(vec3((hl - hr) * k, (hd - hu) * k, 1.0));
    gl_FragColor = vec4(n.xy * 0.5 + 0.5, clamp(s.rough, 0.02, 1.0), s.tint);
  }
}
`;
