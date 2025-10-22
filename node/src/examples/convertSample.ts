import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import { AudioStreamConverter, MuLawStreamDecoder } from '../index';

interface WavInfo {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
  samples: Int16Array;
}

function parseWav(path: string): WavInfo {
  const buffer = readFileSync(path);
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Unsupported WAV header');
  }

  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataSize = 0;

  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    const chunkDataStart = offset + 8;

    if (chunkId === 'fmt ') {
      const audioFormat = buffer.readUInt16LE(chunkDataStart);
      if (audioFormat !== 1) {
        throw new Error('Only PCM WAV files are supported');
      }
      channels = buffer.readUInt16LE(chunkDataStart + 2);
      sampleRate = buffer.readUInt32LE(chunkDataStart + 4);
      bitsPerSample = buffer.readUInt16LE(chunkDataStart + 14);
    } else if (chunkId === 'data') {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
      break;
    }

    offset = chunkDataStart + chunkSize + (chunkSize % 2);
  }

  if (!dataOffset || !sampleRate || bitsPerSample !== 16) {
    throw new Error('Malformed WAV file');
  }

  const totalSamples = dataSize / 2;
  const pcmAll = new Int16Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    pcmAll[i] = buffer.readInt16LE(dataOffset + i * 2);
  }

  let mono: Int16Array;
  if (channels <= 1) {
    mono = pcmAll;
  } else {
    const frames = totalSamples / channels;
    mono = new Int16Array(frames);
    for (let i = 0; i < frames; i++) {
      mono[i] = pcmAll[i * channels];
    }
  }

  return { sampleRate, channels, bitsPerSample, samples: mono };
}

function writeWav(path: string, data: Buffer, sampleRate: number): void {
  const header = Buffer.alloc(44);
  const dataSize = data.length;
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * PCM_BYTES_PER_SAMPLE, 28);
  header.writeUInt16LE(PCM_BYTES_PER_SAMPLE, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);
  writeFileSync(path, Buffer.concat([header, data]));
}

const PCM_BYTES_PER_SAMPLE = 2;

function int16ArrayToBytes(data: Int16Array): Buffer {
  const buffer = Buffer.alloc(data.length * 2);
  for (let i = 0; i < data.length; i++) {
    buffer.writeInt16LE(data[i], i * 2);
  }
  return buffer;
}

function decodeAll(decoder: MuLawStreamDecoder, data: Buffer): Buffer {
  const chunk = decoder.chunkSize * 1;
  const buffers: Buffer[] = [];
  for (let offset = 0; offset < data.length; offset += chunk) {
    const converted = decoder.convertChunk(data.subarray(offset, Math.min(offset + chunk, data.length)));
    if (converted) {
      buffers.push(Buffer.isBuffer(converted) ? converted : Buffer.from(converted));
    }
  }
  const finalChunk = decoder.flush();
  if (finalChunk) {
    buffers.push(Buffer.isBuffer(finalChunk) ? finalChunk : Buffer.from(finalChunk));
  }
  return Buffer.concat(buffers);
}

function ensureTwentyFourK(samples: Int16Array, sampleRate: number): Int16Array {
  if (sampleRate !== 24000) {
    console.warn(`Warning: expected 24000 Hz input, received ${sampleRate} Hz. Continuing without resampling.`);
  }
  return samples;
}

function encodeStream(converter: AudioStreamConverter, samples: Int16Array): Buffer {
  const buffers: Buffer[] = [];
  const chunkSize = converter.chunkSize;
  for (let offset = 0; offset < samples.length; offset += chunkSize) {
    const chunk = samples.subarray(offset, Math.min(offset + chunkSize, samples.length));
    const bytes = int16ArrayToBytes(chunk);
    const converted = converter.convertChunk(bytes);
    if (converted) {
      buffers.push(Buffer.isBuffer(converted) ? converted : Buffer.from(converted));
    }
  }
  const finalChunk = converter.flush();
  if (finalChunk) {
    buffers.push(Buffer.isBuffer(finalChunk) ? finalChunk : Buffer.from(finalChunk));
  }
  return Buffer.concat(buffers);
}

function main(): void {
  const dataDir = resolve(__dirname, '../../src/examples/data');
  const inputWav = resolve(dataDir, 'sample_pcm16_24khz.wav');
  const ulawPath = resolve(dataDir, 'down_sample_8khz.ulaw');
  const downWavPath = resolve(dataDir, 'down_sample_8khz.wav');
  const upWavPath = resolve(dataDir, 'up_sample_pcm16_24khz_decoded.wav');

  const wav = parseWav(inputWav);
  const pcm24 = ensureTwentyFourK(wav.samples, wav.sampleRate);

  const converter = new AudioStreamConverter();
  const ulawData = encodeStream(converter, pcm24);
  writeFileSync(ulawPath, ulawData);

  const ulawBytes = readFileSync(ulawPath);
  const decoder8k = new MuLawStreamDecoder(8000, 8000);
  const decoded8k = decodeAll(decoder8k, ulawBytes);
  writeWav(downWavPath, decoded8k, 8000);

  const decoder24k = new MuLawStreamDecoder(8000, 24000);
  const decoded24k = decodeAll(decoder24k, ulawBytes);
  writeWav(upWavPath, decoded24k, 24000);

  console.log(`Converted ${inputWav} → ${ulawPath}`);
  console.log(`Decoded μ-law to ${downWavPath}`);
  console.log(`Upsampled μ-law to ${upWavPath}`);
}

if (require.main === module) {
  main();
}
