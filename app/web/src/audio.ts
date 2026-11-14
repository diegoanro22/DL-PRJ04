// Envolvente del audio para dibujar la banda de seguridad. Solo se usa en el navegador:
// el audio no se guarda en ningún lado.

export const WINDOW_S = 4 // el modelo usa los primeros 4 s (64,600 muestras a 16 kHz)
export const MAX_RECORD_S = 8

export interface Envolvente {
  /** RMS normalizado (0-1) por intervalo de BIN_S segundos */
  valores: number[]
  duracion: number
}

export const BIN_S = 0.025

export function envolventeDeMuestras(data: Float32Array, sampleRate: number): Envolvente {
  const porBin = Math.max(1, Math.round(sampleRate * BIN_S))
  const valores: number[] = []
  for (let i = 0; i < data.length; i += porBin) {
    let suma = 0
    const fin = Math.min(data.length, i + porBin)
    for (let j = i; j < fin; j++) suma += data[j] * data[j]
    valores.push(Math.sqrt(suma / (fin - i)))
  }
  return { valores: normalizar(valores), duracion: data.length / sampleRate }
}

/** Escala para que el pico (percentil 98) quede en 1, sin que un clic aislado aplaste todo. */
export function normalizar(valores: number[]): number[] {
  if (!valores.length) return valores
  const orden = [...valores].sort((a, b) => a - b)
  const ref = orden[Math.floor(orden.length * 0.98)] || orden[orden.length - 1] || 1
  return valores.map((v) => Math.min(1, v / (ref || 1)))
}

export async function decodificar(blob: Blob): Promise<Envolvente | null> {
  const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  if (!Ctx) return null
  const ctx = new Ctx()
  try {
    const buffer = await ctx.decodeAudioData(await blob.arrayBuffer())
    const canal = buffer.getChannelData(0)
    const maxMuestras = Math.min(canal.length, Math.round(buffer.sampleRate * 10))
    return envolventeDeMuestras(canal.subarray(0, maxMuestras), buffer.sampleRate)
  } catch {
    return null
  } finally {
    void ctx.close()
  }
}

export function formatoSegundos(s: number): string {
  return `${s.toFixed(1)} s`
}
