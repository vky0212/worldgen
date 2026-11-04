/**
 * In-place patching of the tiles' glTF materials for the photo-realistic
 * mode. Materials are never swapped (the analytic colour modes swap them and
 * restore the originals): a hook is installed once per material and the
 * photo-realistic shader variant is switched on and off through the program
 * cache key. Every variant gets the shared uniforms, so switching back and
 * forth never leaves a program without its uniforms.
 */
import { Color, DataTexture, FloatType, InstancedMesh, Mesh, MeshStandardMaterial, NearestFilter, RGBAFormat, Vector3, Vector4 } from 'three';
import type { Material, Object3D, Texture, WebGLProgramParametersWithUniforms } from 'three';
import { STYLE_TEXELS, groupOfMaterial, packStyles, styleFor } from './buildingStyle';
import type { RegionProfile } from './buildingStyle';
import { FRAGMENT_AO, FRAGMENT_EMISSIVE, FRAGMENT_FOG, FRAGMENT_HEAD, FRAGMENT_SURFACE, VERTEX_HEAD, VERTEX_TAIL, VERTEX_TRANSFORM } from './shaders/surface';
import type { QualitySettings } from './quality';
import type { EntityRecord } from '../types';

export type SurfaceKind = 'terrain' | 'road' | 'path' | 'concrete' | 'water' | 'trunk' | 'foliage' | 'building' | 'generic';

export function kindOfMaterial(name: string, layer: string): SurfaceKind {
  if (name === 'terrain' || layer === 'terrain') return 'terrain';
  if (name === 'water') return 'water';
  if (name === 'road_path') return 'path';
  if (name.startsWith('road_') || name === 'bridge_deck') return 'road';
  if (name.startsWith('bridge_') || name.startsWith('tunnel_')) return 'concrete';
  if (name.startsWith('building_')) return 'building';
  if (name === 'tree_trunk') return 'trunk';
  if (name.startsWith('foliage_')) return 'foliage';
  return 'generic';
}

const KIND_DEFINE: Record<SurfaceKind, string> = {
  terrain: 'PR_TERRAIN',
  road: 'PR_ROAD',
  path: 'PR_ROAD\n#define PR_PATH',
  concrete: 'PR_CONCRETE',
  water: 'PR_WATER',
  trunk: 'PR_TRUNK',
  foliage: 'PR_FOLIAGE',
  building: 'PR_BUILDING',
  generic: 'PR_GENERIC',
};

/** Uniforms shared by every patched material. */
export interface SharedUniforms {
  uPrTime: { value: number };
  uPrSunDir: { value: Vector3 };
  uPrSunColor: { value: Color };
  uPrNight: { value: number };
  uPrFogDensity: { value: number };
  uPrFogBase: { value: number };
  uPrFogFalloff: { value: number };
  uPrFogColor: { value: Color };
  uPrDetailFade: { value: number };
  /** Fog colour = environment radiance × this (matches the visible sky dome). */
  uPrFogSky: { value: number };
  uPrAlb: { value: Texture | null };
  uPrSurf: { value: Texture | null };
}

interface HookState {
  kind: SurfaceKind;
  /** Per-material uniforms (building styles, ortho imagery). */
  own: Record<string, { value: unknown }>;
  hasOrtho: boolean;
}

export interface OrthoInfo {
  texture: Texture;
  /** ENU bounds xmin, ymin, xmax, ymax. */
  bounds: [number, number, number, number];
  metresPerPixel: number;
}

export interface PatchContext {
  region: RegionProfile;
  recordFor(id: string): EntityRecord | undefined;
}

export class MaterialPatcher {
  readonly uniforms: SharedUniforms = {
    uPrTime: { value: 0 },
    uPrSunDir: { value: new Vector3(0, 1, 0) },
    uPrSunColor: { value: new Color(1, 1, 1) },
    uPrNight: { value: 0 },
    uPrFogDensity: { value: 0 },
    uPrFogBase: { value: 0 },
    uPrFogFalloff: { value: 300 },
    uPrFogColor: { value: new Color(0.7, 0.8, 0.9) },
    uPrDetailFade: { value: 700 },
    uPrFogSky: { value: 1.6 },
    uPrAlb: { value: null },
    uPrSurf: { value: null },
  };
  active = false;
  private flags = '';
  private settings: QualitySettings | null = null;
  satellite = false;
  private readonly hooked = new Set<MeshStandardMaterial>();
  private readonly states = new WeakMap<Material, HookState>();
  /** Building style textures per mesh. */
  private readonly styleTextures = new Map<Mesh, DataTexture>();

  /** Feature flags that change the shader source. */
  setSettings(s: QualitySettings): void {
    this.settings = s;
    const flags = `f${s.facadeDetail}w${s.waterDetail}${s.wind ? 'W' : ''}${this.satellite ? 'S' : ''}`;
    if (flags !== this.flags) {
      this.flags = flags;
      this.invalidate();
    }
  }

  setSatellite(on: boolean): void {
    if (on === this.satellite) return;
    this.satellite = on;
    if (this.settings) {
      this.flags = '';
      this.setSettings(this.settings);
    }
  }

  setActive(on: boolean): void {
    if (on === this.active) return;
    this.active = on;
    this.invalidate();
  }

  /** Hooks every standard material under `root` (a tile scene). */
  hookTile(root: Object3D, layerOf: (o: Object3D) => string): void {
    root.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        if (!(m instanceof MeshStandardMaterial)) continue;
        this.hook(m, kindOfMaterial(m.name, layerOf(o)));
        if (this.states.get(m)?.kind === 'building' && !Array.isArray(o.material)) this.bindBuildingMesh(o, m);
      }
    });
  }

  /** Builds (or rebuilds after meta.json arrived) the style texture of a building mesh. */
  updateBuildingStyles(mesh: Mesh, featureIds: readonly string[], ctx: PatchContext): void {
    const mat = mesh.material;
    if (!(mat instanceof MeshStandardMaterial)) return;
    const fallback = groupOfMaterial(mat.name);
    const styles = featureIds.map((id) => styleFor(id, ctx.recordFor(id), ctx.region, fallback));
    const n = Math.max(1, styles.length);
    let tex = this.styleTextures.get(mesh);
    if (!tex || tex.image.height !== n) {
      tex?.dispose();
      tex = new DataTexture(new Float32Array(n * STYLE_TEXELS * 4), STYLE_TEXELS, n, RGBAFormat, FloatType);
      tex.minFilter = NearestFilter;
      tex.magFilter = NearestFilter;
      tex.generateMipmaps = false;
      this.styleTextures.set(mesh, tex);
    }
    packStyles(styles, tex.image.data as Float32Array);
    tex.needsUpdate = true;
    const st = this.states.get(mat);
    if (st) st.own.uPrBld.value = tex;
  }

  /** Satellite imagery for a terrain mesh (null removes it). */
  setOrtho(mesh: Mesh, ortho: OrthoInfo | null): void {
    const mat = mesh.material;
    if (!(mat instanceof MeshStandardMaterial)) return;
    const st = this.states.get(mat);
    if (!st || st.kind !== 'terrain') return;
    const had = st.hasOrtho;
    st.hasOrtho = !!ortho;
    if (ortho) {
      // Keep the uniform objects: compiled programs hold references to them.
      st.own.uPrOrtho ??= { value: null };
      st.own.uPrOrthoBounds ??= { value: new Vector4() };
      st.own.uPrOrthoMpp ??= { value: 1 };
      st.own.uPrOrtho.value = ortho.texture;
      (st.own.uPrOrthoBounds.value as Vector4).set(...ortho.bounds);
      st.own.uPrOrthoMpp.value = ortho.metresPerPixel;
    }
    if (had !== st.hasOrtho) mat.needsUpdate = true;
  }

  /** Frees the building style textures of the meshes under a tile root being unloaded. */
  forgetTile(root: Object3D): void {
    root.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      const tex = this.styleTextures.get(o);
      if (!tex) return;
      tex.dispose();
      this.styleTextures.delete(o);
    });
  }

  /** Number of building style textures alive (for leak checks). */
  get styleTextureCount(): number {
    return this.styleTextures.size;
  }

  dispose(): void {
    for (const t of this.styleTextures.values()) t.dispose();
    this.styleTextures.clear();
  }

  private hook(m: MeshStandardMaterial, kind: SurfaceKind): void {
    if (this.states.has(m)) return;
    const state: HookState = { kind, own: kind === 'building' ? { uPrBld: { value: null } } : {}, hasOrtho: false };
    this.states.set(m, state);
    this.hooked.add(m);
    m.addEventListener('dispose', () => this.hooked.delete(m));
    m.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms) => {
      Object.assign(shader.uniforms, this.uniforms, state.own);
      if (!this.active) return;
      this.inject(shader, state);
    };
    m.customProgramCacheKey = () => {
      if (!this.active) return 'pr-off';
      return `pr:${state.kind}:${this.flags}:${state.hasOrtho ? 'o' : ''}`;
    };
    m.needsUpdate = true;
  }

  /** Style textures are per mesh: a building material shared by two meshes gets a copy. */
  private readonly boundTo = new WeakMap<Material, Mesh>();

  private bindBuildingMesh(mesh: Mesh, mat: MeshStandardMaterial): MeshStandardMaterial {
    const owner = this.boundTo.get(mat);
    if (!owner || owner === mesh) {
      this.boundTo.set(mat, mesh);
      return mat;
    }
    const copy = mat.clone();
    mesh.material = copy;
    this.hook(copy, 'building');
    this.boundTo.set(copy, mesh);
    return copy;
  }

  private inject(shader: WebGLProgramParametersWithUniforms, state: HookState): void {
    const s = this.settings;
    const defines = [
      `#define ${KIND_DEFINE[state.kind]}`,
      `#define PR_FACADE ${s?.facadeDetail ?? 1}`,
      `#define PR_WATER_DETAIL ${s?.waterDetail ?? 1}`,
      s?.wind ? '#define PR_WIND' : '',
      state.hasOrtho && this.satellite ? '#define PR_ORTHO' : '',
    ].join('\n');
    // An InstancedMesh shares its material with nothing else; vertex colours come from glTF.
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `${defines}\n#include <common>\n${VERTEX_HEAD}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERTEX_TRANSFORM}`)
      .replace('#include <project_vertex>', `#include <project_vertex>\n${VERTEX_TAIL}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `${defines}\n#include <common>\n${FRAGMENT_HEAD}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>\n${FRAGMENT_SURFACE}`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n${FRAGMENT_EMISSIVE}`)
      .replace('#include <aomap_fragment>', `#include <aomap_fragment>\n${FRAGMENT_AO}`)
      .replace('#include <fog_fragment>', FRAGMENT_FOG);
  }

  private invalidate(): void {
    for (const m of this.hooked) m.needsUpdate = true;
  }
}

/** True for instanced foliage/trunk meshes (tree layer). */
export function isTreeMesh(o: Object3D): o is InstancedMesh {
  return o instanceof InstancedMesh;
}
