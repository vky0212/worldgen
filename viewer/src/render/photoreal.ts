/**
 * Photo-realistic render mode: physical sky and image-based lighting from it,
 * sun at the real solar position for the world's latitude, camera-fitted sun
 * shadows that re-render only when they move, height fog, procedural
 * materials, tree LOD and an HDR post chain (AO, bloom, AgX tone mapping).
 * A frame-rate governor holds the frame rate above 45 fps.
 */
import {
  ACESFilmicToneMapping,
  CircleGeometry,
  Color,
  DepthTexture,
  DirectionalLight,
  HalfFloatType,
  HemisphereLight,
  Mesh,
  MeshBasicMaterial,
  PCFShadowMap,
  PMREMGenerator,
  PerspectiveCamera,
  ClampToEdgeWrapping,
  SRGBColorSpace,
  Scene,
  TextureLoader,
  UnsignedIntType,
  Vector2,
  Vector3,
  WebGLRenderTarget,
} from 'three';
import type { Line, Object3D, Points, ToneMapping, WebGLRenderer } from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { regionProfileAt } from './buildingStyle';
import type { RegionProfile } from './buildingStyle';
import { GpuTimer } from './gpuTimer';
import { MaterialPatcher } from './materials';
import type { OrthoInfo } from './materials';
import { FEATURE_LABELS, FpsGovernor, LEVELS, LEVEL_LABELS, applyShed, higherLevel, lowerLevel, sheddable } from './quality';
import type { GraphicsLevel, LevelChoice, QualitySettings } from './quality';
import { ShadowFitter } from './shadowFit';
import { dayOfYear, solarPosition, sunDirectionEnu } from './sun';
import { TextureEngine } from './textureEngine';
import { TreeLod } from './treeLod';
import type { LoadedTile, WorldModel } from '../world';
import type { LocalBounds, OrthoIndexInfo } from '../types';
import type { Texture } from 'three';

/** Ground under the procedural detail: landcover colours or satellite imagery. */
export type GroundMode = 'procedural' | 'satellite';

export interface PhotorealStats {
  fps: number;
  gpuMs: number | null;
  level: GraphicsLevel;
  choice: LevelChoice;
  scale: number;
  degraded: boolean;
  shed: string[];
  trees: { full: number; canopy: number; impostor: number; hidden: number };
  bakeMs: number;
  shadowUpdates: number;
}

interface SavedState {
  toneMapping: ToneMapping;
  exposure: number;
  pixelRatio: number;
  shadowEnabled: boolean;
  shadowAuto: boolean;
  background: Scene['background'];
  fog: Scene['fog'];
  environment: Scene['environment'];
  hemi: { intensity: number; color: Color; ground: Color };
  sun: { intensity: number; color: Color; castShadow: boolean; position: Vector3; target: Vector3 };
}

/** Radiance scale of the visible sky dome and of the sky used for lighting. */
const SKY_SCALE = 0.2;
const DOME_SCALE = 1.6;
/** Sun : sky irradiance ≈ 6 : 1 on a clear day. */
const SUN_INTENSITY = 6.0;
const ENV_INTENSITY = 0.7;
const DAY_EXPOSURE = 0.42;
const NIGHT_EXPOSURE = 2.4;

export class PhotorealRenderer {
  readonly patcher = new MaterialPatcher();
  readonly governor = new FpsGovernor();
  enabled = false;
  choice: LevelChoice = 'auto';
  level: GraphicsLevel = 'high';
  settings: QualitySettings = LEVELS.high;
  /** Local solar time, hours. */
  timeOfDay = 10.5;
  onStats: ((s: PhotorealStats) => void) | null = null;
  /** GPU memory a tile gained or lost outside its content (orthophoto textures). */
  onTileBytes: ((tileId: string, bytes: number) => void) | null = null;
  /** GPU frame timer (shared with the standard rendering and the benchmark). */
  readonly gpu: GpuTimer;

  private readonly renderer: WebGLRenderer;
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly world: WorldModel;
  private readonly sun: DirectionalLight;
  private readonly hemi: HemisphereLight;
  private readonly textures: TextureEngine;
  private readonly trees = new TreeLod();
  private readonly shadowFit = new ShadowFitter();
  private readonly sky = new Sky();
  private readonly envScene = new Scene();
  private readonly envSky = new Sky();
  private readonly pmrem: PMREMGenerator;
  private envTarget: WebGLRenderTarget | null = null;
  private envDirty = true;
  private envDirtyAt = 0;
  private sceneRT: WebGLRenderTarget | null = null;
  private aoRT: WebGLRenderTarget | null = null;
  private gtao: GTAOPass | null = null;
  private bloom: UnrealBloomPass | null = null;
  private readonly output = new OutputPass();
  private saved: SavedState | null = null;
  private region: RegionProfile = regionProfileAt(Number.NaN, Number.NaN);
  private latitude = 25;
  private readonly doy = dayOfYear(new Date());
  private bounds: LocalBounds = { xmin: -500, ymin: -500, xmax: 500, ymax: 500, zmin: 0, zmax: 50 };
  private readonly sunDir = new Vector3(0, 1, 0);
  private sunElevation = 1;
  private lastFrame = 0;
  private lastTileAddedAt = -Infinity;
  private statsAt = 0;
  private readonly size = new Vector2();
  private time = 0;
  ground: GroundMode = 'procedural';
  private orthoBase: URL | null = null;
  private orthoMpp = 10;
  private readonly orthoTextures = new Map<string, Texture>();
  private readonly orthoLoading = new Set<string>();
  private readonly orthoLoader = new TextureLoader();

  constructor(renderer: WebGLRenderer, scene: Scene, camera: PerspectiveCamera, world: WorldModel, sun: DirectionalLight, hemi: HemisphereLight) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.world = world;
    this.sun = sun;
    this.hemi = hemi;
    this.textures = new TextureEngine(renderer);
    this.gpu = new GpuTimer(renderer.getContext());
    this.pmrem = new PMREMGenerator(renderer);

    this.sky.name = 'photoreal_sky';
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -10;
    this.sky.scale.setScalar(1000);
    this.envSky.scale.setScalar(50);
    this.envScene.add(this.envSky);
    // Ground seen by the environment: dim, vegetated.
    const ground = new Mesh(new CircleGeometry(90, 32), new MeshBasicMaterial({ color: new Color(0.05, 0.06, 0.04) }));
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.5;
    this.envScene.add(ground);
    for (const s of [this.sky, this.envSky]) {
      const u = s.material.uniforms;
      u.turbidity.value = 3.2;
      u.rayleigh.value = 1.4;
      u.mieCoefficient.value = 0.004;
      u.mieDirectionalG.value = 0.8;
      u.cloudCoverage.value = 0.35;
      u.cloudDensity.value = 0.45;
    }
    this.envSky.material.uniforms.showSunDisc.value = 0;
    // Preetham radiance is ~4x a sunlit white surface; real skies are ~1/4 of it.
    // Scale both domes and darken the main dome below the horizon (distant ground haze).
    const patchSky = (sky: Sky, ground: number): void => {
      sky.material.onBeforeCompile = (sh) => {
        sh.uniforms.uSkyScale = { value: sky === this.sky ? SKY_SCALE * DOME_SCALE : SKY_SCALE };
        sh.uniforms.uSkyGround = { value: ground };
        sh.fragmentShader = sh.fragmentShader
          .replace('void main() {', 'uniform float uSkyScale;\nuniform float uSkyGround;\nvoid main() {')
          .replace('gl_FragColor = vec4( texColor, 1.0 );', 'gl_FragColor = vec4( texColor * uSkyScale * mix( uSkyGround, 1.0, smoothstep( -0.12, 0.03, direction.y ) ), 1.0 );');
      };
      sky.material.needsUpdate = true;
    };
    patchSky(this.sky, 0.3);
    patchSky(this.envSky, 1.0);
    document.addEventListener('visibilitychange', () => this.governor.pause(performance.now() + 1500));
  }

  /** Region of the world (manifest centre): building styles and solar position. */
  setLocation(lat: number | null, lon: number | null): void {
    this.region = regionProfileAt(lat ?? Number.NaN, lon ?? Number.NaN);
    if (lat !== null && Number.isFinite(lat)) this.latitude = lat;
    this.updateSun();
  }

  get regionLabel(): string {
    return this.region.label;
  }

  setBounds(b: LocalBounds): void {
    this.bounds = b;
    const u = this.patcher.uniforms;
    u.uPrFogBase.value = b.zmin ?? 0;
    u.uPrFogFalloff.value = Math.max(250, ((b.zmax ?? 0) - (b.zmin ?? 0)) * 0.6);
  }

  setTimeOfDay(hours: number): void {
    this.timeOfDay = hours;
    this.updateSun();
  }

  setLevel(choice: LevelChoice): void {
    this.choice = choice;
    this.applyLevel(choice === 'auto' ? 'high' : choice);
  }

  /** Where `tiles/<id>.ortho.jpg` live, and their effective resolution (index.ortho). */
  setOrthoSource(base: URL, info: OrthoIndexInfo | null | undefined): void {
    this.orthoBase = base;
    this.orthoMpp = Number(info?.effective_source_m_per_px ?? info?.source_m_per_px ?? info?.m_per_px ?? 10) || 10;
    this.orthoAvailable = !!info;
  }

  setGround(mode: GroundMode): void {
    this.ground = mode;
    this.patcher.setSatellite(mode === 'satellite');
    this.loadAllOrtho();
  }

  /** True once the world's tile index lists orthophotos. */
  get hasOrtho(): boolean {
    return this.orthoBase !== null && this.orthoAvailable;
  }

  private orthoAvailable = false;

  private loadAllOrtho(): void {
    if (!this.enabled || this.ground !== 'satellite') return;
    for (const tile of this.world.tiles.values()) this.loadOrtho(tile);
  }

  /** Number of tiles with imagery on the GPU (for the review tooling). */
  get orthoCount(): number {
    return this.orthoTextures.size;
  }

  private loadOrtho(tile: LoadedTile): void {
    const path = tile.entry.ortho;
    const id = tile.entry.id;
    if (!this.enabled || !path || !this.orthoBase || this.orthoTextures.has(id) || this.orthoLoading.has(id)) return;
    this.orthoLoading.add(id);
    // Rebuilds keep the file names: the tile hash keeps a cached image from being reused.
    const url = new URL(path.replace(/^\.?\//, '') + (tile.entry.hash ? `?h=${tile.entry.hash.slice(0, 12)}` : ''), this.orthoBase).href;
    this.orthoLoader.load(
      url,
      (tex) => {
        this.orthoLoading.delete(id);
        tex.colorSpace = SRGBColorSpace;
        tex.wrapS = ClampToEdgeWrapping;
        tex.wrapT = ClampToEdgeWrapping;
        tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
        if (this.world.tiles.get(id) !== tile) {
          // The tile was unloaded while its image was on the way.
          tex.dispose();
          return;
        }
        this.orthoTextures.set(id, tex);
        this.setOrtho(tile, { texture: tex, bounds: tile.entry.bounds, metresPerPixel: this.orthoMpp });
        const img = tex.image as { width?: number; height?: number } | undefined;
        this.onTileBytes?.(id, Math.round((img?.width ?? 0) * (img?.height ?? 0) * 4 * (4 / 3)));
      },
      undefined,
      () => this.orthoLoading.delete(id),
    );
  }

  setEnabled(on: boolean): void {
    if (on === this.enabled) return;
    this.enabled = on;
    if (on) this.enable();
    else this.disable();
  }

  /** A tile GLB arrived. */
  onTileAdded(tile: LoadedTile): void {
    this.collectOverlays(tile.entry.id, tile.root);
    this.patcher.hookTile(tile.root, (o) => this.world.pickableFor(o)?.layer ?? layerOf(o));
    this.trees.addTile(tile.entry.id, tile.entry, tile.layers.get('trees') ?? []);
    this.updateBuildings(tile);
    if (this.ground === 'satellite') this.loadOrtho(tile);
    if (this.enabled) {
      this.applyMeshShadows(tile.root);
      this.shadowFit.invalidate();
      // Loading hitches are not a reason to lower the quality: hold the governor
      // when tiles start arriving, but not for every tile of a continuous stream.
      const now = performance.now();
      if (now - this.lastTileAddedAt > 2000) this.governor.pause(now + 1200);
      this.lastTileAddedAt = now;
    }
  }

  /** A tile is being unloaded: frees what this renderer created for it. */
  onTileRemoved(tile: LoadedTile): void {
    const id = tile.entry.id;
    const tex = this.orthoTextures.get(id);
    if (tex) {
      for (const p of tile.pickables) {
        if (p.layer === 'terrain' && p.object instanceof Mesh) this.patcher.setOrtho(p.object, null);
      }
      tex.dispose();
      this.orthoTextures.delete(id);
    }
    this.orthoLoading.delete(id);
    this.overlays.delete(id);
    this.patcher.forgetTile(tile.root);
    this.trees.removeTile(id);
    this.shadowFit.invalidate();
  }

  /** Resources held per tile, for leak checks (should follow the loaded tiles). */
  resourceCounts(): { ortho: number; overlays: number; styleTextures: number; treeTiles: number } {
    return { ortho: this.orthoTextures.size, overlays: this.overlays.size, styleTextures: this.patcher.styleTextureCount, treeTiles: this.trees.tileCount };
  }

  /** A tile's meta.json arrived: building styles can use the real records now. */
  onMetaLoaded(tileId: string): void {
    const tile = this.world.tiles.get(tileId);
    if (tile) this.updateBuildings(tile);
  }

  setOrtho(tile: LoadedTile, ortho: OrthoInfo | null): void {
    for (const p of tile.pickables) {
      if (p.layer === 'terrain' && p.object instanceof Mesh) this.patcher.setOrtho(p.object, ortho);
    }
  }

  /**
   * Draws a frame. `focus` is what the camera looks at, `viewDistance` how
   * far away it is, `viewRange` how far the camera sees (far plane).
   */
  render(focus: Vector3, viewDistance: number, viewRange: number): void {
    const now = performance.now();
    const frameMs = this.lastFrame ? now - this.lastFrame : 16.7;
    this.lastFrame = now;
    this.time += Math.min(frameMs, 100) / 1000;
    const r = this.renderer;

    this.handleGovernor(now, frameMs);
    if (this.envDirty && now >= this.envDirtyAt) this.rebuildEnvironment();

    const s = this.settings;
    const u = this.patcher.uniforms;
    u.uPrTime.value = this.time;
    // Haze thick enough to soften the far plane, thin when looking from high up.
    u.uPrFogDensity.value = 1 / (Math.max(viewRange, 1000) * 1.5);
    this.sky.position.copy(this.camera.position);
    this.sky.material.uniforms.time.value = this.time;
    if (this.trees.update(this.camera.position, s, now)) this.shadowFit.invalidate();
    this.fitShadow(focus, viewDistance);
    this.ensureTargets();

    this.gpu.begin();
    const sceneRT = this.sceneRT!;
    r.setRenderTarget(sceneRT);
    r.clear();
    r.render(this.scene, this.camera);
    let current = sceneRT;
    if (this.gtao && s.ao) {
      const radius = Math.min(12, Math.max(0.6, viewDistance * 0.012));
      this.gtao.updateGtaoMaterial({ radius, distanceExponent: 1.4, thickness: radius * 0.8, scale: 1.0, samples: s.aoFullRes ? 16 : 10 });
      this.gtao.render(r, this.aoRT!, sceneRT, 0, false);
      current = this.aoRT!;
    }
    if (this.bloom && s.bloom) this.bloom.render(r, null as unknown as WebGLRenderTarget, current, 0, false);
    this.output.renderToScreen = true;
    this.output.render(r, null as unknown as WebGLRenderTarget, current, 0, false);
    this.gpu.end();

    if (now - this.statsAt > 500) {
      this.statsAt = now;
      this.onStats?.(this.stats());
    }
  }

  stats(): PhotorealStats {
    return {
      fps: this.governor.fps,
      gpuMs: this.governor.gpuMs,
      level: this.level,
      choice: this.choice,
      scale: this.governor.scale,
      degraded: this.governor.degraded,
      shed: this.governor.shed.map((f) => FEATURE_LABELS[f]),
      trees: this.trees.counts,
      bakeMs: this.textures.lastBakeMs,
      shadowUpdates: this.shadowFit.updates,
    };
  }

  statusText(): string {
    const st = this.stats();
    const level = this.choice === 'auto' ? `Auto · ${LEVEL_LABELS[st.level]}` : LEVEL_LABELS[st.level];
    return `${Math.round(st.fps)} fps · ${level} · ${st.scale.toFixed(2)}×`;
  }

  // ------------------------------------------------------------ internals

  private enable(): void {
    const r = this.renderer;
    this.saved = {
      toneMapping: r.toneMapping,
      exposure: r.toneMappingExposure,
      pixelRatio: r.getPixelRatio(),
      shadowEnabled: r.shadowMap.enabled,
      shadowAuto: r.shadowMap.autoUpdate,
      background: this.scene.background,
      fog: this.scene.fog,
      environment: this.scene.environment,
      hemi: { intensity: this.hemi.intensity, color: this.hemi.color.clone(), ground: this.hemi.groundColor.clone() },
      sun: {
        intensity: this.sun.intensity,
        color: this.sun.color.clone(),
        castShadow: this.sun.castShadow,
        position: this.sun.position.clone(),
        target: this.sun.target.position.clone(),
      },
    };
    r.toneMapping = ACESFilmicToneMapping;
    r.shadowMap.type = PCFShadowMap;
    r.shadowMap.autoUpdate = false;
    this.scene.background = null;
    this.scene.fog = null;
    this.scene.add(this.sky);
    this.patcher.setActive(true);
    this.applyLevel(this.choice === 'auto' ? this.level : this.choice);
    this.updateSun();
    this.envDirty = true;
    this.envDirtyAt = 0;
    this.applyMeshShadows(this.world.root);
    this.setOverlaysVisible(false);
    this.loadAllOrtho();
    this.shadowFit.invalidate();
    this.lastFrame = 0;
  }

  private disable(): void {
    const r = this.renderer;
    const s = this.saved;
    this.patcher.setActive(false);
    this.scene.remove(this.sky);
    this.trees.reset(false);
    this.setOverlaysVisible(true);
    if (s) {
      r.toneMapping = s.toneMapping;
      r.toneMappingExposure = s.exposure;
      r.setPixelRatio(s.pixelRatio);
      r.shadowMap.enabled = s.shadowEnabled;
      r.shadowMap.autoUpdate = s.shadowAuto;
      this.scene.background = s.background;
      this.scene.fog = s.fog;
      this.scene.environment = s.environment;
      this.hemi.intensity = s.hemi.intensity;
      this.hemi.color.copy(s.hemi.color);
      this.hemi.groundColor.copy(s.hemi.ground);
      this.sun.intensity = s.sun.intensity;
      this.sun.color.copy(s.sun.color);
      this.sun.castShadow = s.sun.castShadow;
      this.sun.position.copy(s.sun.position);
      this.sun.target.position.copy(s.sun.target);
      this.sun.target.updateMatrixWorld();
    }
    this.saved = null;
    this.disposeTargets();
  }

  private applyLevel(level: GraphicsLevel): void {
    this.level = level;
    this.governor.reset(1, sheddable(LEVELS[level]), performance.now());
    this.governor.pause(performance.now() + 2000);
    this.applySettings();
  }

  private applySettings(): void {
    const prev = this.settings;
    const s = applyShed(LEVELS[this.level], this.governor.shed);
    this.settings = s;
    if (!this.enabled) return;
    const r = this.renderer;
    if (this.textures.ensure(s.textureSize, s.anisotropy)) {
      this.patcher.uniforms.uPrAlb.value = this.textures.albedo;
      this.patcher.uniforms.uPrSurf.value = this.textures.surface;
    }
    this.patcher.setSettings(s);
    this.patcher.uniforms.uPrDetailFade.value = s.facadeDetail >= 2 ? 900 : 500;
    r.shadowMap.enabled = s.shadows;
    this.sun.castShadow = s.shadows;
    if (this.sun.shadow.mapSize.x !== s.shadowMapSize) {
      this.sun.shadow.mapSize.set(s.shadowMapSize, s.shadowMapSize);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
    this.sun.shadow.radius = s.shadowMapSize >= 4096 ? 2.5 : 1.5;
    if (prev.msaa !== s.msaa || prev.ao !== s.ao || prev.aoFullRes !== s.aoFullRes || prev.bloom !== s.bloom) this.disposeTargets();
    if (prev.terrainCastsShadow !== s.terrainCastsShadow) this.applyMeshShadows(this.world.root);
    this.trees.update(this.camera.position, s, performance.now(), true);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2) * s.renderScale * this.governor.scale);
    this.shadowFit.invalidate();
  }

  private handleGovernor(now: number, frameMs: number): void {
    const e = this.governor.sample(now, frameMs, this.gpu.poll());
    if (!e) return;
    switch (e.kind) {
      case 'scale':
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2) * this.settings.renderScale * e.scale);
        break;
      case 'shed':
      case 'restore':
        this.applySettings();
        break;
      case 'exhausted':
        if (this.choice === 'auto') {
          const lower = lowerLevel(this.level);
          if (lower) this.applyLevel(lower);
        }
        break;
      case 'surplus':
        if (this.choice === 'auto') {
          const higher = higherLevel(this.level);
          if (higher) {
            this.applyLevel(higher);
            // A level that cannot hold the target falls back on its own (exhausted → lower).
          }
        }
        break;
    }
  }

  private updateSun(): void {
    const p = solarPosition(this.latitude, this.doy, this.timeOfDay);
    const [e, n, up] = sunDirectionEnu(p);
    this.sunDir.set(e, up, -n).normalize();
    this.sunElevation = p.elevation;
    const sinEl = Math.sin(p.elevation);
    const day = smoothstep(-0.04, 0.12, sinEl);
    const warm = smoothstep(0.0, 0.45, sinEl);
    const u = this.patcher.uniforms;
    u.uPrSunDir.value.copy(this.sunDir);
    u.uPrSunColor.value.setRGB(1.0, 0.5 + 0.47 * warm, 0.28 + 0.62 * warm).multiplyScalar(day);
    u.uPrNight.value = 1 - smoothstep(-0.1, 0.05, sinEl);
    for (const s of [this.sky, this.envSky]) s.material.uniforms.sunPosition.value.copy(this.sunDir);
    if (this.enabled) {
      this.sun.color.copy(u.uPrSunColor.value).multiplyScalar(1 / Math.max(day, 1e-3));
      this.sun.intensity = SUN_INTENSITY * day;
      this.hemi.color.setRGB(0.45, 0.55, 0.8);
      this.hemi.groundColor.setRGB(0.1, 0.1, 0.12);
      this.hemi.intensity = 0.06 + 0.25 * (1 - day);
      this.renderer.toneMappingExposure = DAY_EXPOSURE + (NIGHT_EXPOSURE - DAY_EXPOSURE) * (1 - day);
      this.envDirty = true;
      this.envDirtyAt = performance.now() + 120;
      this.shadowFit.invalidate();
    }
  }

  private rebuildEnvironment(): void {
    this.envDirty = false;
    const prev = this.envTarget;
    this.envTarget = this.pmrem.fromScene(this.envScene, 0.02, 0.1, 200);
    this.scene.environment = this.envTarget.texture;
    this.scene.environmentIntensity = ENV_INTENSITY;
    prev?.dispose();
  }

  /** Points the sun's shadow camera at what the camera looks at; re-renders the map only when it moved. */
  private fitShadow(focus: Vector3, viewDistance: number): void {
    const s = this.settings;
    const light = this.sun;
    if (!s.shadows || this.sunElevation < 0.02) {
      this.renderer.shadowMap.enabled = false;
      return;
    }
    this.renderer.shadowMap.enabled = true;
    const zSpan = (this.bounds.zmax ?? 0) - (this.bounds.zmin ?? 0);
    if (this.shadowFit.fit(light, this.sunDir, focus, viewDistance, { maxHalf: s.shadowMaxHalf, mapSize: s.shadowMapSize, zSpan })) {
      this.renderer.shadowMap.needsUpdate = true;
    }
  }

  private ensureTargets(): void {
    const r = this.renderer;
    r.getDrawingBufferSize(this.size);
    const w = Math.max(1, Math.floor(this.size.x));
    const h = Math.max(1, Math.floor(this.size.y));
    const s = this.settings;
    if (this.sceneRT && this.sceneRT.width === w && this.sceneRT.height === h) return;
    if (this.sceneRT) {
      this.sceneRT.setSize(w, h);
      this.aoRT?.setSize(w, h);
      this.gtao?.setSize(s.aoFullRes ? w : Math.ceil(w / 2), s.aoFullRes ? h : Math.ceil(h / 2));
      this.bloom?.setSize(Math.ceil(w / 2), Math.ceil(h / 2));
      return;
    }
    const depth = new DepthTexture(w, h, UnsignedIntType);
    this.sceneRT = new WebGLRenderTarget(w, h, { type: HalfFloatType, samples: s.msaa, depthTexture: depth });
    this.sceneRT.texture.name = 'photoreal.scene';
    if (s.ao) {
      this.aoRT = new WebGLRenderTarget(w, h, { type: HalfFloatType, depthBuffer: false });
      const aw = s.aoFullRes ? w : Math.ceil(w / 2);
      const ah = s.aoFullRes ? h : Math.ceil(h / 2);
      this.gtao = new GTAOPass(this.scene, this.camera, aw, ah, { depthTexture: depth });
      this.gtao.blendIntensity = 0.85;
      this.gtao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 6, rings: 2, samples: 12 });
    }
    if (s.bloom) {
      this.bloom = new UnrealBloomPass(new Vector2(Math.ceil(w / 2), Math.ceil(h / 2)), 0.12, 0.4, 3.0);
    }
  }

  private disposeTargets(): void {
    this.sceneRT?.depthTexture?.dispose();
    this.sceneRT?.dispose();
    this.aoRT?.dispose();
    this.gtao?.dispose();
    this.bloom?.dispose();
    this.sceneRT = null;
    this.aoRT = null;
    this.gtao = null;
    this.bloom = null;
  }

  private updateBuildings(tile: LoadedTile): void {
    const ctx = { region: this.region, recordFor: (id: string) => this.world.recordFor(id, tile.entry.id) };
    for (const p of tile.pickables) {
      if (p.layer !== 'buildings' || !(p.object instanceof Mesh) || !p.featureIds) continue;
      this.patcher.updateBuildingStyles(p.object, p.featureIds, ctx);
    }
  }

  /** Data-visualisation lines in tiles (vegetation region outlines), per tile: hidden in photo mode. */
  private readonly overlays = new Map<string, Object3D[]>();

  private collectOverlays(tileId: string, root: Object3D): void {
    const list: Object3D[] = [];
    root.traverse((o) => {
      if ((o as Line).isLine || (o as Points).isPoints) {
        list.push(o);
        if (this.enabled) o.visible = false;
      }
    });
    if (list.length) this.overlays.set(tileId, list);
  }

  private setOverlaysVisible(v: boolean): void {
    for (const list of this.overlays.values()) for (const o of list) o.visible = v;
  }

  private applyMeshShadows(root: Object3D): void {
    const s = this.settings;
    root.traverse((o) => {
      if (!(o instanceof Mesh) || o.userData.lodOf) return;
      const layer = this.world.pickableFor(o)?.layer ?? layerOf(o);
      if (layer === 'trees') return; // the tree LOD decides
      o.receiveShadow = true;
      o.castShadow = layer === 'terrain' ? s.terrainCastsShadow : layer !== 'water' && layer !== 'vegetation_regions';
    });
  }
}

function layerOf(o: Object3D): string {
  for (let p: Object3D | null = o; p; p = p.parent) {
    const l = p.userData?.layer;
    if (typeof l === 'string') return l;
  }
  return '';
}

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

