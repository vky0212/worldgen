/**
 * Shader patches injected into the glTF MeshStandardMaterials in
 * photo-realistic mode (via onBeforeCompile). One source with #ifdef per
 * material kind; the kind and the feature flags are part of the program
 * cache key, so each combination compiles once and is shared by all tiles.
 */
import { BAY_WIDTH, LAYER, ROOF_TILE_M } from '../buildingStyle';
import { NOISE_GLSL } from './noise';

const layerDefines = Object.entries(LAYER)
  .map(([k, v]) => `#define L_${k} ${v.toFixed(1)}`)
  .join('\n');

export const SURFACE_COMMON = /* glsl */ `
${layerDefines}
varying vec3 vPrWorld;
varying vec3 vPrNormalW;
varying vec2 vPrUv;
uniform float uPrTime;
uniform vec3 uPrSunDir;
uniform vec3 uPrSunColor;
uniform float uPrNight;
uniform float uPrFogDensity;
uniform float uPrFogBase;
uniform float uPrFogFalloff;
uniform vec3 uPrFogColor;
uniform float uPrDetailFade;
uniform float uPrFogSky;
${NOISE_GLSL}
`;

export const VERTEX_HEAD = /* glsl */ `
${SURFACE_COMMON}
#if defined(PR_FOLIAGE) || defined(PR_TRUNK)
varying vec3 vPrTint;
varying float vPrLocalY;
#endif
#ifdef PR_BUILDING
attribute float _feature_id_0;
uniform highp sampler2D uPrBld;
flat varying vec4 vPrS0;
flat varying vec4 vPrS1;
flat varying vec4 vPrS2;
flat varying vec4 vPrS3;
#endif
`;

/** After <begin_vertex>: wind on trees (in model units: trees are unit-height). */
export const VERTEX_TRANSFORM = /* glsl */ `
#if defined(PR_FOLIAGE) || defined(PR_TRUNK)
  #ifdef USE_INSTANCING
    vec3 prIp = instanceMatrix[3].xyz;
  #else
    vec3 prIp = vec3(0.0);
  #endif
  float prH1 = prHash13(floor(prIp * 0.5) + 0.37);
  float prH2 = prHash13(floor(prIp * 0.5) + 7.1);
  vPrTint = mix(vec3(0.8, 1.0, 0.74), vec3(1.18, 1.04, 0.7), prH1) * (0.72 + 0.56 * prH2);
  vPrLocalY = position.y;
  #ifdef PR_WIND
    float prSway = sin(uPrTime * 1.3 + prIp.x * 0.05 + prIp.z * 0.07) * 0.018 + sin(uPrTime * 2.9 + prIp.x * 0.13) * 0.006;
    float prBend = max(transformed.y, 0.0);
    transformed.x += prSway * prBend * prBend;
    transformed.z += prSway * 0.6 * prBend * prBend;
  #endif
#endif
`;

/** After <project_vertex>: world position/normal, building style fetch. */
export const VERTEX_TAIL = /* glsl */ `
{
  vec4 prW = vec4(transformed, 1.0);
  vec3 prNo = objectNormal;
  #ifdef USE_INSTANCING
    prW = instanceMatrix * prW;
    prNo = mat3(instanceMatrix) * prNo;
  #endif
  prW = modelMatrix * prW;
  vPrWorld = prW.xyz;
  vPrNormalW = normalize(mat3(modelMatrix) * prNo);
  vPrUv = uv;
}
#ifdef PR_BUILDING
{
  int prFid = int(_feature_id_0 + 0.5);
  vPrS0 = texelFetch(uPrBld, ivec2(0, prFid), 0);
  vPrS1 = texelFetch(uPrBld, ivec2(1, prFid), 0);
  vPrS2 = texelFetch(uPrBld, ivec2(2, prFid), 0);
  vPrS3 = texelFetch(uPrBld, ivec2(3, prFid), 0);
}
#endif
`;

const bayTable = `const float PR_BAY[${BAY_WIDTH.length}] = float[${BAY_WIDTH.length}](${BAY_WIDTH.map((v) => v.toFixed(2)).join(', ')});`;
const roofTable = `const float PR_ROOF[${ROOF_TILE_M.length}] = float[${ROOF_TILE_M.length}](${ROOF_TILE_M.map((v) => v.toFixed(2)).join(', ')});`;

export const FRAGMENT_HEAD = /* glsl */ `
${SURFACE_COMMON}
uniform highp sampler2DArray uPrAlb;
uniform highp sampler2DArray uPrSurf;
#if defined(PR_FOLIAGE) || defined(PR_TRUNK)
varying vec3 vPrTint;
varying float vPrLocalY;
#endif
#ifdef PR_BUILDING
flat varying vec4 vPrS0;
flat varying vec4 vPrS1;
flat varying vec4 vPrS2;
flat varying vec4 vPrS3;
${bayTable}
${roofTable}
#endif
#ifdef PR_ORTHO
uniform sampler2D uPrOrtho;
uniform vec4 uPrOrthoBounds;
uniform float uPrOrthoMpp;
#endif
vec3 prViewN(vec3 nW) { return normalize((viewMatrix * vec4(nW, 0.0)).xyz); }
vec3 prDecodeN(vec4 s, float strength) {
  vec2 xy = (s.xy * 2.0 - 1.0) * strength;
  return vec3(xy, sqrt(max(0.0, 1.0 - dot(xy, xy))));
}
float prLum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// The exporter's colours are display (sRGB) colours stored as linear factors.
vec3 prToLinear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec4 prTri(float layer, vec3 p, vec3 w, float scale, bool surf) {
  vec4 x = surf ? texture(uPrSurf, vec3(p.zy * scale, layer)) : texture(uPrAlb, vec3(p.zy * scale, layer));
  vec4 y = surf ? texture(uPrSurf, vec3(p.xz * scale, layer)) : texture(uPrAlb, vec3(p.xz * scale, layer));
  vec4 z = surf ? texture(uPrSurf, vec3(p.xy * scale, layer)) : texture(uPrAlb, vec3(p.xy * scale, layer));
  return x * w.x + y * w.y + z * w.z;
}
vec3 prTriNormal(float layer, vec3 p, vec3 n, vec3 w, float scale, float strength) {
  // Whiteout-style blend of the three planar normal maps.
  vec3 tx = prDecodeN(texture(uPrSurf, vec3(p.zy * scale, layer)), strength);
  vec3 ty = prDecodeN(texture(uPrSurf, vec3(p.xz * scale, layer)), strength);
  vec3 tz = prDecodeN(texture(uPrSurf, vec3(p.xy * scale, layer)), strength);
  vec3 nx = vec3(0.0, tx.y, tx.x);
  vec3 ny = vec3(ty.x, 0.0, ty.y);
  vec3 nz = vec3(tz.x, tz.y, 0.0);
  return normalize(n + nx * w.x + ny * w.y + nz * w.z);
}
vec3 prTriWeights(vec3 n) {
  vec3 w = pow(abs(n), vec3(4.0));
  return w / max(w.x + w.y + w.z, 1e-4);
}
`;

/** Replaces <normal_fragment_maps> (keeps it): surface appearance per kind. */
export const FRAGMENT_SURFACE = /* glsl */ `
vec3 prEmissive = vec3(0.0);
float prAO = 1.0;
vec3 prN = normalize(vPrNormalW) * faceDirection;
vec3 prP = vPrWorld;
float prDist = length(prP - cameraPosition);
float prNear = 1.0 - smoothstep(uPrDetailFade * 0.3, uPrDetailFade, prDist);

#ifdef PR_TERRAIN
{
  vec3 base = prToLinear(diffuseColor.rgb);
  float macro = prFbm(prP.xz * 0.011);
  float macro2 = prNoise(prP.xz * 0.073 + 3.1);
  #ifdef PR_ORTHO
  {
    vec2 ouv = vec2((prP.x - uPrOrthoBounds.x) / (uPrOrthoBounds.z - uPrOrthoBounds.x),
                    ((-prP.z) - uPrOrthoBounds.y) / (uPrOrthoBounds.w - uPrOrthoBounds.y));
    vec3 sat = texture(uPrOrtho, ouv).rgb;
    // Satellite pixels are shown as they are from afar and blend into the
    // landcover colour + procedural detail where they would be magnified.
    float pxPerTexel = uPrOrthoMpp / max(length(fwidth(prP.xz)), 1e-4);
    float satW = 1.0 - smoothstep(2.0, 8.0, pxPerTexel) * 0.55;
    // Lift baked shadows (much darker than the landcover colour).
    float lift = clamp(prLum(base) * 0.6 / max(prLum(sat), 1e-3), 1.0, 1.6);
    base = mix(base, sat * lift, satW);
  }
  #endif
  float green = clamp((base.g - max(base.r, base.b)) * 8.0, 0.0, 1.0);
  // Landcover greens are yellowish display colours: pull vegetation towards chlorophyll green.
  base *= mix(vec3(1.0), vec3(0.86, 1.0, 0.95), green);
  float lum = prLum(base);
  float forestish = green * (1.0 - smoothstep(0.05, 0.12, lum));
  vec2 uvA = prP.xz * 0.25;
  vec2 uvB = mat2(0.8, -0.6, 0.6, 0.8) * prP.xz * 0.061;
  vec3 grassD = mix(texture(uPrAlb, vec3(uvA, L_GRASS)).rgb, texture(uPrAlb, vec3(uvB, L_GRASS)).rgb, 0.5);
  vec3 forestD = texture(uPrAlb, vec3(uvA * 0.8, L_FOREST)).rgb;
  vec3 soilD = texture(uPrAlb, vec3(uvA, L_SOIL)).rgb;
  vec3 grassAvg = textureLod(uPrAlb, vec3(0.5, 0.5, L_GRASS), 12.0).rgb;
  vec3 forestAvg = textureLod(uPrAlb, vec3(0.5, 0.5, L_FOREST), 12.0).rgb;
  vec3 soilAvg = textureLod(uPrAlb, vec3(0.5, 0.5, L_SOIL), 12.0).rgb;
  vec3 veg = mix(grassD / max(grassAvg, vec3(1e-3)), forestD / max(forestAvg, vec3(1e-3)), forestish);
  // Bare / built-up ground: soil when brownish, concrete when grey.
  float sat = (max(base.r, max(base.g, base.b)) - min(base.r, min(base.g, base.b))) / max(max(base.r, max(base.g, base.b)), 1e-3);
  vec3 concD = texture(uPrAlb, vec3(uvA, L_CONCRETE_PLAIN)).rgb;
  vec3 concAvg = textureLod(uPrAlb, vec3(0.5, 0.5, L_CONCRETE_PLAIN), 12.0).rgb;
  vec3 bare = mix(concD / max(concAvg, vec3(1e-3)), soilD / max(soilAvg, vec3(1e-3)), smoothstep(0.08, 0.25, sat));
  vec3 ratio = mix(bare, veg, green);
  vec3 albedo = base * mix(vec3(1.0), ratio, 0.7 * prNear + 0.25) * (0.86 + 0.28 * macro);
  vec4 surfG = texture(uPrSurf, vec3(uvA, green > 0.5 ? L_GRASS : L_SOIL));
  float rough = 0.95;
  vec3 tn = prDecodeN(surfG, 0.8 * prNear);
  vec3 nW = normalize(prN + vec3(tn.x, 0.0, tn.y));
  float slope = 1.0 - prN.y;
  float rockW = smoothstep(0.26, 0.48, slope + (macro2 - 0.5) * 0.22);
  if (rockW > 0.01) {
    vec3 w = prTriWeights(prN);
    vec3 rock = prTri(L_ROCK, prP, w, 1.0 / 7.0, false).rgb;
    rock = mix(rock, rock * base / max(lum, 1e-3) * 0.35, 0.3) * (0.85 + 0.3 * macro);
    albedo = mix(albedo, rock, rockW);
    rough = mix(rough, 0.85, rockW);
    nW = normalize(mix(nW, prTriNormal(L_ROCK, prP, prN, w, 1.0 / 7.0, 1.2 * prNear + 0.3), rockW));
  }
  diffuseColor.rgb = albedo;
  roughnessFactor = rough;
  metalnessFactor = 0.0;
  normal = prViewN(nW);
  prAO = (0.78 + 0.22 * macro) * (1.0 - forestish * 0.25);
}
#endif

#ifdef PR_ROAD
{
  float deckSide = step(prN.y, 0.6);
  vec2 uvR = prP.xz / 3.5;
  #ifdef PR_PATH
    float layer = L_GRAVEL;
    float k = 0.85;
  #else
    float layer = L_ASPHALT;
    float k = prLum(diffuseColor.rgb) > 0.26 ? 1.25 : 1.0;
  #endif
  vec3 tang = normalize(vec3(-prN.z, 0.0, prN.x) + 1e-5);
  vec2 uvS = vec2(dot(prP.xz, tang.xz), prP.y) / 3.0;
  float L = mix(layer, L_CONCRETE_PLAIN, deckSide);
  vec2 uv = mix(uvR, uvS, deckSide);
  vec4 a = texture(uPrAlb, vec3(uv, L));
  vec4 s = texture(uPrSurf, vec3(uv, L));
  float wear = prFbm(prP.xz * 0.05);
  diffuseColor.rgb = a.rgb * k * (0.85 + 0.3 * wear);
  roughnessFactor = s.b;
  metalnessFactor = 0.0;
  vec3 tn = prDecodeN(s, 0.7 * prNear);
  vec3 nW = deckSide > 0.5 ? prN : normalize(prN + vec3(tn.x, 0.0, tn.y));
  normal = prViewN(nW);
}
#endif

#ifdef PR_CONCRETE
{
  vec3 w = prTriWeights(prN);
  vec3 c = prTri(L_CONCRETE_PLAIN, prP, w, 1.0 / 3.0, false).rgb;
  diffuseColor.rgb = c * prToLinear(diffuseColor.rgb) / 0.3;
  roughnessFactor = 0.85;
  metalnessFactor = 0.0;
  normal = prViewN(prTriNormal(L_CONCRETE_PLAIN, prP, prN, w, 1.0 / 3.0, prNear));
  prAO = 0.9;
}
#endif

#ifdef PR_WATER
{
  float t = uPrTime;
  vec2 p = prP.xz;
  vec2 g = vec2(0.0);
  vec2 d1 = vec2(0.8, 0.6); vec2 d2 = vec2(-0.45, 0.9); vec2 d3 = vec2(0.95, -0.3); vec2 d4 = vec2(-0.7, -0.7);
  g += 0.06 * 0.9 * d1 * cos(dot(d1, p) * 0.9 + t * 1.1);
  g += 0.04 * 1.7 * d2 * cos(dot(d2, p) * 1.7 + t * 1.6);
  #if PR_WATER_DETAIL >= 2
  g += 0.02 * 3.1 * d3 * cos(dot(d3, p) * 3.1 + t * 2.3);
  g += 0.012 * 5.3 * d4 * cos(dot(d4, p) * 5.3 + t * 3.1);
  #endif
  #if PR_WATER_DETAIL >= 3
  {
    vec2 q = p * 2.3 + vec2(t * 0.35, -t * 0.27);
    float e = 0.05;
    float n0 = prNoise(q);
    g += vec2(prNoise(q + vec2(e, 0.0)) - n0, prNoise(q + vec2(0.0, e)) - n0) / e * 0.025;
  }
  #endif
  g *= mix(0.35, 1.0, prNear);
  vec3 nW = normalize(vec3(-g.x, 1.0, -g.y));
  normal = prViewN(nW);
  vec3 V = normalize(cameraPosition - prP);
  float fres = pow(1.0 - max(dot(V, nW), 0.0), 5.0);
  diffuseColor.rgb = vec3(0.012, 0.04, 0.045);
  diffuseColor.a = mix(0.8, 0.98, fres);
  #if PR_WATER_DETAIL >= 2
  roughnessFactor = 0.03;
  #else
  roughnessFactor = 0.09;
  #endif
  metalnessFactor = 0.0;
}
#endif

#ifdef PR_TRUNK
{
  float n = prNoise(vec2(prP.y * 3.0, (prP.x + prP.z) * 9.0));
  diffuseColor.rgb = prToLinear(diffuseColor.rgb) * (0.7 + 0.5 * n) * vPrTint.g;
  roughnessFactor = 0.95;
  metalnessFactor = 0.0;
  prAO = 0.55;
}
#endif

#ifdef PR_FOLIAGE
{
  float n1 = prNoise(prP.xz * 1.7 + prP.y * 1.3);
  float n2 = prNoise(prP.xz * 4.1 - prP.y * 2.7);
  vec3 alb = prToLinear(diffuseColor.rgb) * 1.1 * vPrTint * (0.8 + 0.4 * n1);
  alb = mix(vec3(prLum(alb)), alb, 0.62) * vec3(1.05, 0.95, 0.9);
  diffuseColor.rgb = alb;
  vec3 nW = normalize(prN + vec3(n1 - 0.5, (n2 - 0.5) * 0.5, n2 - 0.5) * 0.9 * (0.4 + 0.6 * prNear));
  normal = prViewN(nW);
  roughnessFactor = 0.82;
  metalnessFactor = 0.0;
  prAO = mix(0.45, 1.0, smoothstep(0.05, 0.9, vPrLocalY));
  vec3 V = normalize(prP - cameraPosition);
  float back = pow(max(dot(V, uPrSunDir), 0.0), 4.0);
  prEmissive += uPrSunColor * alb * back * 0.6 * (1.0 - uPrNight);
}
#endif

#ifdef PR_BUILDING
{
  float wallL = vPrS0.x;
  float groundL = vPrS0.y;
  float roofL = vPrS0.z;
  float fh = max(vPrS0.w, 2.0);
  vec3 wallTint = vPrS1.rgb;
  float seed = vPrS1.a;
  vec3 roofTint = vPrS2.rgb;
  float roofGloss = vPrS2.a;
  float gz = vPrS3.x;
  float lit = vPrS3.y;
  float hAbove = prP.y - gz;
  bool isWall = abs(prN.y) < 0.5;
  vec3 albedo;
  float rough;
  float metal = 0.0;
  vec3 nW = prN;
  if (isWall) {
    float u = vPrUv.x * 4.0;
    float storey = max(floor(hAbove / fh), 0.0);
    float L = storey < 0.5 ? groundL : wallL;
    float bw = PR_BAY[int(L + 0.5)];
    vec2 g = vec2(u / bw, hAbove / fh);
    vec2 gdx = dFdx(g);
    vec2 gdy = dFdy(g);
    vec2 tuv = fract(g);
    vec3 T = normalize(vec3(prN.z, 0.0, -prN.x));
    vec3 B = vec3(0.0, 1.0, 0.0);
    #if PR_FACADE >= 3
    {
      // One-step parallax: recessed glass shifts with the view.
      vec4 probe = textureGrad(uPrAlb, vec3(tuv, L), gdx, gdy);
      if (probe.a > 0.75) {
        vec3 V = normalize(cameraPosition - prP);
        vec2 vt = vec2(dot(V, T), dot(V, B)) / max(dot(V, prN), 0.25);
        tuv -= vt * vec2(0.18 / bw, 0.18 / fh) * 0.35;
      }
    }
    #endif
    vec4 a = textureGrad(uPrAlb, vec3(tuv, L), gdx, gdy);
    vec4 s = textureGrad(uPrSurf, vec3(tuv, L), gdx, gdy);
    float glass = a.a;
    // Weathering: real facades are darker than fresh paint.
    albedo = mix(a.rgb, a.rgb * wallTint / 0.86, s.a) * (0.78 + 0.1 * prNoise(vec2(u, hAbove) * 0.15 + seed * 40.0));
    rough = s.b;
    vec2 win = vec2(floor(g.x), storey);
    float hw = prHash12(win + seed * 91.7);
    float hw2 = prHash12(win * 1.7 + seed * 13.1 + 5.0);
    if (glass > 0.9) {
      if (wallL == L_CURTAIN && storey > 0.5) {
        albedo = wallTint * 0.55;
        metal = 0.75;
        rough = 0.03 + 0.03 * hw;
      } else {
        // Glass: reflective (coated, F0 ~0.1), with interiors and curtains behind it.
        vec3 interior = mix(vec3(0.03, 0.035, 0.04), vec3(0.42, 0.38, 0.3) * 0.45, step(0.55, hw) * (0.5 + 0.5 * hw2));
        albedo = mix(interior, vec3(0.55, 0.62, 0.66), 0.12);
        metal = 0.1;
        rough = 0.03 + 0.05 * hw2;
      }
      #if PR_FACADE < 2
      rough = 0.25;
      #endif
      float on = step(hw2, lit) * uPrNight;
      prEmissive += vec3(1.0, 0.72, 0.42) * (1.2 + 1.6 * hw) * on;
      prAO *= 0.9;
    } else if (L == L_CURTAIN && tuv.y > 0.035 && tuv.y < 0.195 && tuv.x > 0.03 && tuv.x < 0.97) {
      // Opaque spandrel glass.
      albedo = wallTint * 0.12;
      metal = 0.6;
      rough = 0.06;
    } else if (L == L_SHOPFRONT && tuv.y > 0.825 && tuv.y < 0.955 && tuv.x > 0.035 && tuv.x < 0.965) {
      // Shop sign: colour per bay.
      vec3 sign = 0.5 + 0.45 * cos(6.2831 * (hw + vec3(0.0, 0.33, 0.67)));
      albedo = sign * 0.7;
      rough = 0.4;
      prEmissive += sign * 0.8 * uPrNight * step(0.4, hw2);
    }
    #if PR_FACADE >= 1
    vec3 tn = prDecodeN(s, 1.0);
    nW = normalize(T * tn.x + B * tn.y + prN * tn.z);
    #endif
    prAO *= mix(0.55, 1.0, smoothstep(0.0, 2.5, hAbove));
  } else {
    // Roof (or a flat underside): tiles run along the eaves.
    vec2 hz = prN.xz;
    float slopeLen = length(hz);
    vec2 d = slopeLen > 0.08 ? hz / slopeLen : vec2(0.0, 1.0);
    vec2 pr = vec2(dot(prP.xz, vec2(-d.y, d.x)), dot(prP.xz, d));
    float tile = PR_ROOF[int(clamp(roofL - L_ROOF_CLAY, 0.0, 4.0) + 0.5)];
    vec2 ruv = pr / tile;
    vec4 a = texture(uPrAlb, vec3(ruv, roofL));
    vec4 s = texture(uPrSurf, vec3(ruv, roofL));
    albedo = mix(a.rgb, a.rgb * roofTint / 0.8, s.a);
    rough = mix(s.b, s.b * 0.4, roofGloss);
    metal = roofL == L_ROOF_METAL ? 0.55 : 0.0;
    #if PR_FACADE >= 1
    vec3 T = normalize(vec3(-d.y, 0.0, d.x));
    vec3 B = normalize(cross(prN, T));
    vec3 tn = prDecodeN(s, 1.0);
    nW = normalize(T * tn.x - B * tn.y + prN * tn.z);
    #endif
    if (prN.y < -0.5) { albedo = vec3(0.3); nW = prN; }
  }
  diffuseColor.rgb = albedo;
  roughnessFactor = rough;
  metalnessFactor = metal;
  normal = prViewN(nW);
}
#endif
`;

/** After <emissivemap_fragment>. */
export const FRAGMENT_EMISSIVE = /* glsl */ `
totalEmissiveRadiance += prEmissive;
`;

/** After <aomap_fragment>: cheap baked-style occlusion of indirect light. */
export const FRAGMENT_AO = /* glsl */ `
reflectedLight.indirectDiffuse *= prAO;
reflectedLight.indirectSpecular *= mix(1.0, prAO, 0.6);
`;

/** Replaces <fog_fragment>: exponential height fog lit by the sky. */
export const FRAGMENT_FOG = /* glsl */ `
if (uPrFogDensity > 0.0) {
  vec3 prV = vPrWorld - cameraPosition;
  float prD = length(prV);
  vec3 prDir = prV / max(prD, 1e-3);
  float k = 1.0 / max(uPrFogFalloff, 1.0);
  float h0 = max(cameraPosition.y - uPrFogBase, 0.0);
  float h1 = max(vPrWorld.y - uPrFogBase, 0.0);
  float dh = h1 - h0;
  float avg = abs(dh) > 0.5 ? (exp(-k * h0) - exp(-k * h1)) / (k * dh) : exp(-k * h0);
  float fogAmt = 1.0 - exp(-uPrFogDensity * prD * avg);
  vec3 fogCol = uPrFogColor;
  #if defined(USE_ENVMAP) && defined(ENVMAP_TYPE_CUBE_UV)
    vec3 fd = normalize(vec3(prDir.x, max(prDir.y, 0.0) * 0.5 + 0.03, prDir.z));
    fogCol = textureCubeUV(envMap, envMapRotation * fd, 0.5).rgb * uPrFogSky;
  #endif
  fogCol += uPrSunColor * pow(max(dot(prDir, uPrSunDir), 0.0), 8.0) * 0.08;
  gl_FragColor.rgb = mix(gl_FragColor.rgb, fogCol, clamp(fogAmt, 0.0, 1.0));
}
`;
