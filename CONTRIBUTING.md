# Contributing

Thank you for contributing to Sypra.

## Local setup

Requirements: Go 1.23+, Node 20+, and Corepack/pnpm.

```powershell
$env:GOCACHE = "$PWD\.cache\go-build"
go test ./...

Set-Location web
corepack pnpm install --frozen-lockfile
corepack pnpm run build
```

Use `.env.example` as the starting point for local configuration. Do not
commit credentials, local workspace data, model weights, generated binaries,
or user documents.

## Pull requests

- Keep each change focused and include tests for changed behavior.
- Format Go with `gofmt` and follow existing TypeScript conventions.
- Keep outbound network access explicit and configurable.
- Do not commit credentials, model files, workspace data, or generated output.

Contributions are licensed under Apache-2.0.
