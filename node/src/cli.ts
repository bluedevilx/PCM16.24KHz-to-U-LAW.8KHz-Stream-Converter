#!/usr/bin/env node
import { existsSync, mkdirSync } from 'fs';
import { dirname, resolve } from 'path';
import { AudioStreamConverter, MuLawStreamDecoder, convertFile, convertUlawFile } from './index';

function printHelp(): void {
  console.log(`audio-stream-converter (Node.js)

Usage:
  audio-convert <input> <output> [options]

Options:
  --direction <pcm-to-ulaw|ulaw-to-pcm>   Conversion direction (default: pcm-to-ulaw)
  --chunk-samples <number>               Override chunk size in samples
  --input-rate <number>                  Input sample rate (default depends on direction)
  --output-rate <number>                 Output sample rate (default depends on direction)
  --help                                 Show this message
`);
}

interface ParsedArgs {
  [key: string]: string | boolean | string[];
  _: string[];
}

function parseArgs(argv: string[]): ParsedArgs {
  const args: ParsedArgs = { _: [] };
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token === '--help' || token === '-h') {
      args.help = true;
    } else if (token.startsWith('--')) {
      const key = token.slice(2);
      const value = argv[i + 1];
      if (value === undefined || value.startsWith('--')) {
        throw new Error(`Missing value for option ${token}`);
      }
      args[key] = value;
      i++;
    } else {
      positional.push(token);
    }
  }
  args._ = positional;
  return args;
}

export function main(argv = process.argv.slice(2)): void {
  let args: ParsedArgs;
  try {
    args = parseArgs(argv);
  } catch (error) {
    console.error((error as Error).message);
    printHelp();
    process.exit(1);
    return;
  }

  const help = typeof args.help === 'boolean' ? args.help : false;
  if (help || args._.length < 2) {
    printHelp();
    process.exit(help ? 0 : 1);
    return;
  }

  const [inputPath, outputPath] = args._.map((p) => resolve(p));

  const directionArg = args.direction;
  const direction = typeof directionArg === 'string' ? directionArg : 'pcm-to-ulaw';

  const chunkSamplesArg = args['chunk-samples'];
  const chunkSamples = typeof chunkSamplesArg === 'string' ? parseInt(chunkSamplesArg, 10) : undefined;

  const inputRateArg = args['input-rate'];
  const inputRate = typeof inputRateArg === 'string' ? parseInt(inputRateArg, 10) : undefined;

  const outputRateArg = args['output-rate'];
  const outputRate = typeof outputRateArg === 'string' ? parseInt(outputRateArg, 10) : undefined;

  if (!existsSync(dirname(outputPath))) {
    mkdirSync(dirname(outputPath), { recursive: true });
  }

  if (direction === 'pcm-to-ulaw') {
    const converter = new AudioStreamConverter(
      inputRate ?? 24000,
      outputRate ?? 8000,
      chunkSamples ?? 4800,
    );
    convertFile(inputPath, outputPath, { converter, chunkSamples });
  } else if (direction === 'ulaw-to-pcm') {
    const decoder = new MuLawStreamDecoder(
      inputRate ?? 8000,
      outputRate ?? 24000,
      chunkSamples ?? 1600,
    );
    convertUlawFile(inputPath, outputPath, { decoder, chunkSamples });
  } else {
    console.error(`Unknown direction: ${direction}`);
    printHelp();
    process.exit(1);
    return;
  }
}

if (require.main === module) {
  main();
}
