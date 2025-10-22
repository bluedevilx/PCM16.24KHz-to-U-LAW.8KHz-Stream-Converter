# audio-stream-converter (Node.js)

TypeScript implementation of the PCM16 24kHz → μ-law 8kHz streaming converter with
optional reverse conversion, designed for use as a library or CLI tool.

## Features

- Incremental conversion with buffer management (forward and reverse directions)
- Dependency-free implementation (no native bindings)
- CLI compatible with the Python tooling (`npm run cli -- --help`)
- Sample script mirroring the Python `examples/convert_sample.py`
- Strict TypeScript typings and generated declaration files

## Getting Started

```bash
npm install
yarn build # or npm run build
node dist/cli.js --help
```

See `examples/convertSample.ts` for an end-to-end demonstration.
