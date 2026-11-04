/**
 * GPU frame time from EXT_disjoint_timer_query_webgl2 (Chromium; Safari has
 * none, then poll() always returns null). At vsync the frame interval cannot
 * show headroom; the GPU time can.
 */
interface TimerExt {
  TIME_ELAPSED_EXT: number;
  GPU_DISJOINT_EXT: number;
}

export class GpuTimer {
  private readonly gl: WebGL2RenderingContext | null;
  private readonly ext: TimerExt | null;
  private readonly pending: WebGLQuery[] = [];
  private readonly free: WebGLQuery[] = [];
  private active: WebGLQuery | null = null;
  private last: number | null = null;

  constructor(gl: WebGLRenderingContext | WebGL2RenderingContext) {
    const gl2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext ? gl : null;
    this.gl = gl2;
    this.ext = gl2 ? (gl2.getExtension('EXT_disjoint_timer_query_webgl2') as TimerExt | null) : null;
  }

  get available(): boolean {
    return this.ext !== null;
  }

  begin(): void {
    const gl = this.gl;
    if (!gl || !this.ext || this.active || this.pending.length > 4) return;
    const q = this.free.pop() ?? gl.createQuery();
    if (!q) return;
    gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
    this.active = q;
  }

  end(): void {
    const gl = this.gl;
    if (!gl || !this.ext || !this.active) return;
    gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this.pending.push(this.active);
    this.active = null;
  }

  /** Latest finished measurement (ms), or null. */
  poll(): number | null {
    const gl = this.gl;
    if (!gl || !this.ext) return null;
    const disjoint = gl.getParameter(this.ext.GPU_DISJOINT_EXT) as boolean;
    let result: number | null = null;
    while (this.pending.length) {
      const q = this.pending[0];
      if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
      this.pending.shift();
      this.free.push(q);
      if (!disjoint) result = ns / 1e6;
    }
    if (result !== null) this.last = result;
    return result;
  }

  get lastMs(): number | null {
    return this.last;
  }
}
