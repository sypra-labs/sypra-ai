import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const dataPath = resolve(root, '.cache', 'e2e-state.json').replaceAll('\\', '/');

export default defineConfig({
  testDir: './e2e',
  timeout: 45_000,
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: 'http://127.0.0.1:5173', trace: 'retain-on-failure' },
  webServer: [
	{
	  command: 'go run ./cmd/sypra-mock --listen 127.0.0.1:8082',
	  cwd: root,
	  url: 'http://127.0.0.1:8082/v1/models',
	  reuseExistingServer: !process.env.CI,
	},
    {
      command: 'go run ./cmd/sypra serve',
      cwd: root,
      url: 'http://127.0.0.1:8080/readyz',
      reuseExistingServer: !process.env.CI,
      env: {
        ...process.env,
        SYPRA_LISTEN_ADDR: '127.0.0.1:8080',
        SYPRA_CORS_ORIGIN: 'http://127.0.0.1:5173',
        SYPRA_DATA_PATH: dataPath,
        SYPRA_DATABASE_URL: `sqlite://${dataPath.replace(/\.json$/, '.db')}`,
        SYPRA_MODEL_ENDPOINT: '',
      },
    },
    {
      command: 'pnpm run dev -- --host 127.0.0.1',
      cwd: resolve(root, 'web'),
      url: 'http://127.0.0.1:5173',
      reuseExistingServer: !process.env.CI,
    },
  ],
});
