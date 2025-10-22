import { openSync, closeSync, readSync, writeSync } from 'fs';

const PCM_BYTES_PER_SAMPLE = 2;
const ULAW_BYTES_PER_SAMPLE = 1;
const ULAW_BIAS = 0x84;
const FIR_COEFFS = new Float32Array([
  -0.00000000000000000048,
  -0.00050757580902427435,
  -0.00071716151433065534,
  0.00000000000000000124,
  0.00127583090215921402,
  0.00163632573094218969,
  -0.00000000000000000233,
  -0.00255169020965695381,
  -0.00312116579152643681,
  0.00000000000000000374,
  0.00452642282471060753,
  0.00538274692371487617,
  -0.00000000000000000539,
  -0.00746728852391242981,
  -0.00872911326587200165,
  0.00000000000000000719,
  0.01180763635784387589,
  0.01369067002087831497,
  -0.00000000000000000897,
  -0.01839848794043064117,
  -0.02138582989573478699,
  0.00000000000000001059,
  0.02934103831648826599,
  0.03484134003520011902,
  -0.00000000000000001188,
  -0.05182809010148048401,
  -0.06626323610544204712,
  0.00000000000000001271,
  0.13655221462249755859,
  0.27514773607254028320,
  0.33353540301322937012,
  0.27514773607254028320,
  0.13655221462249755859,
  0.00000000000000001271,
  -0.06626323610544204712,
  -0.05182809010148048401,
  -0.00000000000000001188,
  0.03484134003520011902,
  0.02934103831648826599,
  0.00000000000000001059,
  -0.02138582989573478699,
  -0.01839848794043064117,
  -0.00000000000000000897,
  0.01369067002087831497,
  0.01180763635784387589,
  0.00000000000000000719,
  -0.00872911326587200165,
  -0.00746728852391242981,
  -0.00000000000000000539,
  0.00538274692371487617,
  0.00452642282471060753,
  0.00000000000000000374,
  -0.00312116579152643681,
  -0.00255169020965695381,
  -0.00000000000000000233,
  0.00163632573094218969,
  0.00127583090215921402,
  0.00000000000000000124,
  -0.00071716151433065534,
  -0.00050757580902427435,
  -0.00000000000000000048,
]);

const PCM_TO_ULAW = new Uint8Array(65536);
const ULAW_TO_PCM = new Int16Array(256);

(function buildLookupTables() {
  for (let i = 0; i < 65536; i++) {
    const sample = i - 32768;
    PCM_TO_ULAW[i] = encodeSample(sample);
  }
  for (let i = 0; i < 256; i++) {
    ULAW_TO_PCM[i] = decodeSample(i);
  }
})();

function encodeSample(sample: number): number {
  let sign = 0;
  let magnitude = sample;
  if (sample < 0) {
    sign = 0x80;
    magnitude = -sample;
  }
  magnitude = Math.min(magnitude + ULAW_BIAS, 32635);
  let exponent = 7;
  for (let exp = 7; exp >= 0; exp--) {
    if (magnitude & (0x80 << exp)) {
      exponent = exp;
      break;
    }
  }
  const mantissa = (magnitude >> (exponent + 3)) & 0x0f;
  const ulaw = ~(sign | (exponent << 4) | mantissa) & 0xff;
  return ulaw;
}

function decodeSample(ulawByte: number): number {
  let u = ulawByte ^ 0xff;
  const sign = u & 0x80;
  const exponent = (u >> 4) & 0x07;
  const mantissa = u & 0x0f;
  let sample = ((mantissa << 3) + ULAW_BIAS) << exponent;
  sample -= ULAW_BIAS;
  if (sign !== 0) {
    sample = -sample;
  }
  if (sample > 32767) return 32767;
  if (sample < -32768) return -32768;
  return sample;
}

function clampInt16(value: number): number {
  if (value > 32767) return 32767;
  if (value < -32768) return -32768;
  return value | 0;
}

function bytesToInt16LE(bytes: Uint8Array): Int16Array {
  const length = Math.floor(bytes.length / 2);
  const result = new Int16Array(length);
  for (let i = 0; i < length; i++) {
    result[i] = bytes[i * 2] | (bytes[i * 2 + 1] << 8);
  }
  return result;
}

function int16ToBytes(samples: Int16Array): Buffer {
  const buffer = Buffer.allocUnsafe(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    buffer.writeInt16LE(samples[i], i * 2);
  }
  return buffer;
}

function encodePCM16ToULaw(pcm: Int16Array): Buffer {
  const bytes = Buffer.allocUnsafe(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    bytes[i] = PCM_TO_ULAW[pcm[i] + 32768];
  }
  return bytes;
}

function decodeULawToPCM16(data: Uint8Array): Int16Array {
  const result = new Int16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = ULAW_TO_PCM[data[i]];
  }
  return result;
}

function appendInt16Arrays(a: Int16Array, b: Int16Array): Int16Array {
  if (a.length === 0) return Int16Array.from(b);
  if (b.length === 0) return Int16Array.from(a);
  const combined = new Int16Array(a.length + b.length);
  combined.set(a, 0);
  combined.set(b, a.length);
  return combined;
}

function floatFromInt16(samples: Int16Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = samples[i];
  }
  return out;
}

function applyFIR(input: Float32Array, filter: Float32Array): Float32Array {
  const filterLen = filter.length;
  if (input.length < filterLen) {
    return new Float32Array(0);
  }
  const outputLen = input.length - filterLen + 1;
  const output = new Float32Array(outputLen);
  for (let i = 0; i < outputLen; i++) {
    let acc = 0;
    for (let k = 0; k < filterLen; k++) {
      acc += filter[k] * input[i + k];
    }
    output[i] = acc;
  }
  return output;
}

function downsampleTo8k(samples: Int16Array): Int16Array {
  const filtered = applyFIR(floatFromInt16(samples), FIR_COEFFS);
  const downFactor = 3;
  const outputLen = Math.floor(filtered.length / downFactor);
  const output = new Int16Array(outputLen);
  for (let i = 0; i < outputLen; i++) {
    output[i] = clampInt16(Math.round(filtered[i * downFactor]));
  }
  return output;
}

function upsampleTo24k(samples: Int16Array): Int16Array {
  const upFactor = 3;
  const upsampled = new Float32Array(samples.length * upFactor);
  for (let i = 0; i < samples.length; i++) {
    upsampled[i * upFactor] = samples[i];
  }
  const filtered = applyFIR(upsampled, FIR_COEFFS);
  const output = new Int16Array(filtered.length);
  for (let i = 0; i < filtered.length; i++) {
    output[i] = clampInt16(Math.round(filtered[i]));
  }
  return output;
}

export class AudioStreamConverter {
  private residualSamples = new Int16Array(0);
  private byteRemainder: number | null = null;
  private readonly minChunkSize: number;
  private readonly tailInputSize: number;
  private readonly tailOutputSize: number;
  private readonly downFactor: number;

  constructor(
    public readonly inputRate = 24000,
    public readonly outputRate = 8000,
    public readonly chunkSize = 4800,
  ) {
    if (inputRate <= 0 || outputRate <= 0 || chunkSize <= 0) {
      throw new Error('AudioStreamConverter: rates and chunkSize must be positive');
    }
    this.downFactor = Math.floor(inputRate / outputRate);
    this.minChunkSize = Math.max(1, Math.floor(this.inputRate / 100));
    this.tailInputSize = FIR_COEFFS.length;
    this.tailOutputSize = Math.ceil(this.tailInputSize * (this.outputRate / this.inputRate)) + this.downFactor;
  }

  convertChunk(data: Uint8Array | Buffer): Buffer | null {
    if (data.length === 0 && this.residualSamples.length === 0) {
      return null;
    }

    let bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
    if (this.byteRemainder !== null) {
      bytes = Buffer.concat([Buffer.from([this.byteRemainder]), bytes]);
      this.byteRemainder = null;
    }
    if (bytes.length % 2 === 1) {
      this.byteRemainder = bytes[bytes.length - 1];
      bytes = bytes.slice(0, bytes.length - 1);
    }
    if (bytes.length === 0) {
      return null;
    }

    const samples = appendInt16Arrays(this.residualSamples, bytesToInt16LE(bytes));
    if (samples.length < this.minChunkSize) {
      this.residualSamples = Int16Array.from(samples);
      return null;
    }

    const resampled = downsampleTo8k(samples);
    const tailInputLen = Math.min(samples.length, this.tailInputSize);
    const tailOutputLen = Math.min(resampled.length, this.tailOutputSize);
    const emitLen = resampled.length - tailOutputLen;
    if (emitLen <= 0) {
      this.residualSamples = Int16Array.from(samples);
      return null;
    }

    const emitSamples = Int16Array.from(resampled.subarray(0, emitLen));
    this.residualSamples = tailInputLen > 0
      ? Int16Array.from(samples.subarray(samples.length - tailInputLen))
      : new Int16Array(0);
    return encodePCM16ToULaw(emitSamples);
  }

  convertStream(source: Iterable<Uint8Array | Buffer>): Iterable<Buffer> {
    const self = this;
    function* generator() {
      for (const chunk of source) {
        const converted = self.convertChunk(chunk);
        if (converted) {
          yield converted;
        }
      }
    }
    return generator();
  }

  flush(): Buffer | null {
    if (this.byteRemainder !== null) {
      this.byteRemainder = null;
    }
    if (this.residualSamples.length === 0) {
      return null;
    }
    const resampled = downsampleTo8k(this.residualSamples);
    this.residualSamples = new Int16Array(0);
    if (resampled.length === 0) {
      return null;
    }
    return encodePCM16ToULaw(resampled);
  }

  reset(): void {
    this.residualSamples = new Int16Array(0);
    this.byteRemainder = null;
  }
}

export class MuLawStreamDecoder {
  private residualSamples = new Int16Array(0);
  private readonly minChunkSize: number;
  private readonly tailInputSize: number;
  private readonly tailOutputSize: number;
  private readonly upFactor: number;

  constructor(
    public readonly inputRate = 8000,
    public readonly outputRate = 24000,
    public readonly chunkSize = 1600,
  ) {
    if (inputRate <= 0 || outputRate <= 0 || chunkSize <= 0) {
      throw new Error('MuLawStreamDecoder: rates and chunkSize must be positive');
    }
    this.upFactor = Math.floor(outputRate / inputRate);
    this.minChunkSize = Math.max(1, Math.floor(this.inputRate / 100));
    this.tailInputSize = FIR_COEFFS.length;
    this.tailOutputSize = Math.ceil(this.tailInputSize * (this.outputRate / this.inputRate)) + this.upFactor;
  }

  convertChunk(data: Uint8Array | Buffer): Buffer | null {
    if (data.length === 0 && this.residualSamples.length === 0) {
      return null;
    }
    const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const decoded = decodeULawToPCM16(bytes);
    const samples = appendInt16Arrays(this.residualSamples, decoded);
    if (samples.length < this.minChunkSize) {
      this.residualSamples = Int16Array.from(samples);
      return null;
    }

    const resampled = upsampleTo24k(samples);
    const tailInputLen = Math.min(samples.length, this.tailInputSize);
    const tailOutputLen = Math.min(resampled.length, this.tailOutputSize);
    const emitLen = resampled.length - tailOutputLen;
    if (emitLen <= 0) {
      this.residualSamples = Int16Array.from(samples);
      return null;
    }

    const emitSamples = Int16Array.from(resampled.subarray(0, emitLen));
    this.residualSamples = tailInputLen > 0
      ? Int16Array.from(samples.subarray(samples.length - tailInputLen))
      : new Int16Array(0);
    return int16ToBytes(emitSamples);
  }

  convertStream(source: Iterable<Uint8Array | Buffer>): Iterable<Buffer> {
    const self = this;
    function* generator() {
      for (const chunk of source) {
        const converted = self.convertChunk(chunk);
        if (converted) {
          yield converted;
        }
      }
    }
    return generator();
  }

  flush(): Buffer | null {
    if (this.residualSamples.length === 0) {
      return null;
    }
    const resampled = upsampleTo24k(this.residualSamples);
    this.residualSamples = new Int16Array(0);
    if (resampled.length === 0) {
      return null;
    }
    return int16ToBytes(resampled);
  }

  reset(): void {
    this.residualSamples = new Int16Array(0);
  }
}

export function convertFile(
  inputPath: string,
  outputPath: string,
  options: {
    chunkSamples?: number;
    converter?: AudioStreamConverter;
  } = {},
): void {
  const converter = options.converter ?? new AudioStreamConverter();
  converter.reset();

  const chunkSamples = options.chunkSamples ?? converter.chunkSize;
  if (chunkSamples <= 0) {
    throw new Error('chunkSamples must be positive');
  }
  const chunkBytes = chunkSamples * PCM_BYTES_PER_SAMPLE;

  const fdIn = openSync(inputPath, 'r');
  const fdOut = openSync(outputPath, 'w');
  try {
    const buffer = Buffer.allocUnsafe(chunkBytes);
    let bytesRead: number;
    do {
      bytesRead = readSync(fdIn, buffer, 0, chunkBytes, null);
      if (bytesRead > 0) {
        const converted = converter.convertChunk(buffer.subarray(0, bytesRead));
        if (converted) {
          writeSync(fdOut, converted);
        }
      }
    } while (bytesRead === chunkBytes);

    const finalChunk = converter.flush();
    if (finalChunk) {
      writeSync(fdOut, finalChunk);
    }
  } finally {
    closeSync(fdIn);
    closeSync(fdOut);
  }
}

export function convertUlawFile(
  inputPath: string,
  outputPath: string,
  options: {
    chunkSamples?: number;
    decoder?: MuLawStreamDecoder;
  } = {},
): void {
  const decoder = options.decoder ?? new MuLawStreamDecoder();
  decoder.reset();

  const chunkSamples = options.chunkSamples ?? decoder.chunkSize;
  if (chunkSamples <= 0) {
    throw new Error('chunkSamples must be positive');
  }
  const chunkBytes = chunkSamples * ULAW_BYTES_PER_SAMPLE;

  const fdIn = openSync(inputPath, 'r');
  const fdOut = openSync(outputPath, 'w');
  try {
    const buffer = Buffer.allocUnsafe(chunkBytes);
    let bytesRead: number;
    do {
      bytesRead = readSync(fdIn, buffer, 0, chunkBytes, null);
      if (bytesRead > 0) {
        const converted = decoder.convertChunk(buffer.subarray(0, bytesRead));
        if (converted) {
          writeSync(fdOut, converted);
        }
      }
    } while (bytesRead === chunkBytes);

    const finalChunk = decoder.flush();
    if (finalChunk) {
      writeSync(fdOut, finalChunk);
    }
  } finally {
    closeSync(fdIn);
    closeSync(fdOut);
  }
}
