import { createReadStream, existsSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import path from 'node:path';
import { defineConfig } from 'vite';
import type { Connect, Plugin } from 'vite';

/**
 * Serves `/world/*` from the directory named by the WORLDGEN_WORLD environment
 * variable (absolute, or relative to the directory npm was invoked from) in
 * both `vite` (dev) and `vite preview`. Without the variable, `/world/*`
 * answers 404 with a hint instead of falling through to index.html.
 */
function worldgenWorld(): Plugin {
  const raw = process.env.WORLDGEN_WORLD?.trim();
  const baseDir = process.env.INIT_CWD ?? process.cwd();
  const root = raw ? path.resolve(baseDir, raw) : null;

  const types: Record<string, string> = {
    '.json': 'application/json; charset=utf-8',
    '.glb': 'model/gltf-binary',
    '.gltf': 'model/gltf+json',
    '.txt': 'text/plain; charset=utf-8',
    '.tif': 'image/tiff',
    '.tiff': 'image/tiff',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.bin': 'application/octet-stream',
    '.f32': 'application/octet-stream',
    '.parquet': 'application/vnd.apache.parquet',
  };

  const handler: Connect.NextHandleFunction = (req: IncomingMessage, res: ServerResponse) => {
    const send = (status: number, body: string): void => {
      res.statusCode = status;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end(body);
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') return send(405, 'Method not allowed');
    if (!root) {
      return send(404, 'No world directory configured. Start the server with WORLDGEN_WORLD=<path to compiled world>.');
    }
    let rel: string;
    try {
      rel = decodeURIComponent((req.url ?? '/').split('?')[0]);
    } catch {
      return send(400, 'Bad request');
    }
    const file = path.resolve(root, `.${path.posix.normalize(`/${rel}`)}`);
    if (file !== root && !file.startsWith(root + path.sep)) return send(403, 'Forbidden');
    let stat;
    try {
      stat = statSync(file);
    } catch {
      return send(404, `Not found: ${rel}`);
    }
    if (!stat.isFile()) return send(404, `Not a file: ${rel}`);
    res.statusCode = 200;
    res.setHeader('Content-Type', types[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Content-Length', String(stat.size));
    res.setHeader('Cache-Control', 'no-cache');
    if (req.method === 'HEAD') return void res.end();
    createReadStream(file)
      .on('error', () => {
        if (!res.headersSent) send(500, 'Read error');
        else res.destroy();
      })
      .pipe(res);
  };

  const announce = (log: (msg: string) => void): void => {
    if (!root) log('[worldgen] WORLDGEN_WORLD not set: /world/ will return 404 (or pass ?world=<url>).');
    else if (!existsSync(path.join(root, 'manifest.json'))) log(`[worldgen] serving /world/ from ${root} (warning: no manifest.json there)`);
    else log(`[worldgen] serving /world/ from ${root}`);
  };

  return {
    name: 'worldgen-world',
    configureServer(server) {
      announce((m) => server.config.logger.info(m));
      server.middlewares.use('/world', handler);
    },
    configurePreviewServer(server) {
      announce((m) => server.config.logger.info(m));
      server.middlewares.use('/world', handler);
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [worldgenWorld()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
    sourcemap: true,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 5173,
  },
});
