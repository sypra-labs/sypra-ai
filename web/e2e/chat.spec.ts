import { expect, test } from '@playwright/test';

test('creates a chat and renders a streamed external mock response', async ({ page, request }) => {
  const registered = await request.post('http://127.0.0.1:8080/api/v1/models', {
    data: { name: 'Sypra Mock', endpoint: 'http://127.0.0.1:8082/v1', family: 'generic', format: 'api', modalities: ['text'], kind: 'base' },
  });
  expect(registered.ok()).toBeTruthy();
  await page.goto('/');
  await expect(page.getByText('Online')).toBeVisible();
  const composer = page.getByLabel('Message');
  await composer.fill('browser E2E validation');
  await page.getByLabel('Send message').click();
  await expect(page.getByText('browser E2E validation')).toBeVisible();
  await expect(page.getByText('Test response: the mock model received your prompt: browser E2E validation')).toBeVisible({ timeout: 25_000 });
});
