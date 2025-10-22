# Audio Stream Converter Monorepo

This repository contains both the original Python implementation and a TypeScript/Node.js port
of the PCM16 24 kHz ↔ μ-law 8 kHz streaming audio converter.

```
python/  # Python package, CLI, tests, and examples
node/    # TypeScript library/CLI with matching sample conversion
```

- **Python**: see [`python/README.md`](python/README.md) for usage, installation, and examples.
- **Node.js**: see [`node/README.md`](node/README.md) for npm commands, CLI usage, and the sample script.

Both implementations share the same FIR filter, μ-law lookup tables, and conversion semantics so
that artifacts and latencies align across platforms.
