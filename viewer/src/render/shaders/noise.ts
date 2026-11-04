/** Shared GLSL helpers: hashes, value noise, fBm, periodic variants, Voronoi. */
export const NOISE_GLSL = /* glsl */ `
float prHash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float prHash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
vec2 prHash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
vec3 prHash32(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yzz) * p3.zyx);
}
float prNoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(prHash12(i), prHash12(i + vec2(1.0, 0.0)), u.x),
             mix(prHash12(i + vec2(0.0, 1.0)), prHash12(i + vec2(1.0, 1.0)), u.x), u.y);
}
float prFbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * prNoise(p);
    p = p * 2.03 + vec2(17.1, 9.2);
    a *= 0.5;
  }
  return v / 0.9375;
}
`;

/** Periodic (tileable) noise for the texture bake: period in cells. */
export const PERIODIC_NOISE_GLSL = /* glsl */ `
float pnHash(vec2 i, vec2 period, float seed) {
  i = mod(i, period);
  return prHash13(vec3(i, seed));
}
float pNoise(vec2 p, vec2 period, float seed) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(pnHash(i, period, seed), pnHash(i + vec2(1.0, 0.0), period, seed), u.x),
             mix(pnHash(i + vec2(0.0, 1.0), period, seed), pnHash(i + vec2(1.0, 1.0), period, seed), u.x), u.y);
}
// uv in [0,1): tileable fBm with 'base' cells at the first octave.
float pFbm(vec2 uv, vec2 base, float seed) {
  float v = 0.0;
  float a = 0.5;
  vec2 per = base;
  for (int i = 0; i < 5; i++) {
    v += a * pNoise(uv * per, per, seed + float(i) * 7.13);
    per *= 2.0;
    a *= 0.5;
  }
  return v / 0.96875;
}
// Tileable Voronoi: x = distance to nearest, y = distance to 2nd nearest, z = cell id hash.
vec3 pVoronoi(vec2 uv, vec2 cells, float seed) {
  vec2 p = uv * cells;
  vec2 i = floor(p);
  vec2 f = fract(p);
  float d1 = 8.0;
  float d2 = 8.0;
  float id = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 g = vec2(float(x), float(y));
      vec2 c = mod(i + g, cells);
      vec2 o = prHash22(c + seed * 13.7);
      float d = length(g + o - f);
      if (d < d1) { d2 = d1; d1 = d; id = prHash13(vec3(c, seed)); }
      else if (d < d2) { d2 = d; }
    }
  }
  return vec3(d1, d2, id);
}
`;
