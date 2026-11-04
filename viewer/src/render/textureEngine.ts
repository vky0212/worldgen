/**
 * Procedural texture engine: bakes every material layer (building facades,
 * roofs, ground detail) on the GPU into two mipmapped 2D texture arrays,
 * once per graphics level. No image assets are downloaded.
 *
 *  - albedo: sRGB rgb + glass mask (a)
 *  - surface: tangent-space normal xy, roughness, tint mask
 */
import {
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
  SRGBColorSpace,
  ShaderMaterial,
  NoColorSpace,
  WebGLArrayRenderTarget,
} from 'three';
import type { Texture, WebGLRenderer } from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { LAYER_COUNT } from './buildingStyle';
import { BAKE_FRAGMENT, BAKE_VERTEX } from './shaders/bake';

export class TextureEngine {
  private readonly renderer: WebGLRenderer;
  private albedoRT: WebGLArrayRenderTarget | null = null;
  private surfaceRT: WebGLArrayRenderTarget | null = null;
  private material: ShaderMaterial | null = null;
  private quad: FullScreenQuad | null = null;
  size = 0;
  anisotropy = 1;
  /** Milliseconds the last bake took (CPU side, including the GPU flush). */
  lastBakeMs = 0;
  /** Bumped on every bake, so materials know their textures changed. */
  version = 0;

  constructor(renderer: WebGLRenderer) {
    this.renderer = renderer;
  }

  get albedo(): Texture | null {
    return this.albedoRT?.texture ?? null;
  }

  get surface(): Texture | null {
    return this.surfaceRT?.texture ?? null;
  }

  /** (Re)bakes all layers at `size` px when the size or anisotropy changed. Returns true when it baked. */
  ensure(size: number, anisotropy: number): boolean {
    const aniso = Math.max(1, Math.min(anisotropy, this.renderer.capabilities.getMaxAnisotropy()));
    if (this.albedoRT && this.size === size && this.anisotropy === aniso) return false;
    const t0 = performance.now();
    this.disposeTargets();
    this.size = size;
    this.anisotropy = aniso;
    this.albedoRT = this.makeTarget(size, true, aniso);
    this.surfaceRT = this.makeTarget(size, false, aniso);
    this.material ??= new ShaderMaterial({
      vertexShader: BAKE_VERTEX,
      fragmentShader: BAKE_FRAGMENT,
      uniforms: { uLayer: { value: 0 }, uOutput: { value: 0 }, uTexel: { value: 1 / size } },
      depthTest: false,
      depthWrite: false,
    });
    this.quad ??= new FullScreenQuad(this.material);
    this.material.uniforms.uTexel.value = 1 / size;

    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevXr = r.xr.enabled;
    r.xr.enabled = false;
    for (const [rt, output] of [[this.albedoRT, 0], [this.surfaceRT, 1]] as const) {
      this.material.uniforms.uOutput.value = output;
      for (let layer = 0; layer < LAYER_COUNT; layer++) {
        // Mipmaps once, after the last layer (generateMipmap covers the whole array).
        rt.texture.generateMipmaps = layer === LAYER_COUNT - 1;
        this.material.uniforms.uLayer.value = layer;
        r.setRenderTarget(rt, layer);
        this.quad.render(r);
      }
      rt.texture.generateMipmaps = true;
    }
    r.setRenderTarget(prevTarget);
    r.xr.enabled = prevXr;
    this.lastBakeMs = performance.now() - t0;
    this.version++;
    return true;
  }

  dispose(): void {
    this.disposeTargets();
    this.material?.dispose();
    this.quad?.dispose();
    this.material = null;
    this.quad = null;
  }

  private makeTarget(size: number, srgb: boolean, anisotropy: number): WebGLArrayRenderTarget {
    const rt = new WebGLArrayRenderTarget(size, size, LAYER_COUNT, { depthBuffer: false });
    const t = rt.texture;
    t.generateMipmaps = true;
    t.minFilter = LinearMipmapLinearFilter;
    t.magFilter = LinearFilter;
    t.wrapS = RepeatWrapping;
    t.wrapT = RepeatWrapping;
    t.anisotropy = anisotropy;
    t.colorSpace = srgb ? SRGBColorSpace : NoColorSpace;
    return rt;
  }

  private disposeTargets(): void {
    this.albedoRT?.dispose();
    this.surfaceRT?.dispose();
    this.albedoRT = null;
    this.surfaceRT = null;
  }
}
