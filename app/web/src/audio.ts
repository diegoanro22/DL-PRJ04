// Envolvente del audio para dibujar la banda de seguridad. Solo se usa en el navegador:
// el audio no se guarda en ningún lado.

export const WINDOW_S = 64600 / 16000 // una ventana del modelo: 64,600 muestras a 16 kHz (~4 s)
export const MAX_S = 20 // la API analiza hasta 20 s, en ventanas de 4 s
export const MAX_RECORD_S = MAX_S

export interface Tramo {
  inicio: number
  fin: number
}

/** Las mismas ventanas que arma la API (app/inference.py: windows), para dibujarlas antes del resultado. */
export function ventanasDe(duracion: number): Tramo[] {
  const d = Math.min(duracion, MAX_S)
  if (d <= WINDOW_S) return [{ inicio: 0, fin: d }]
  const out: Tramo[] = []
  for (let s = 0; s + WINDOW_S <= d + 1e-9; s += WINDOW_S) out.push({ inicio: s, fin: s + WINDOW_S })
  const ultimo = out[out.length - 1].fin
  if (ultimo < d - 1e-3) out.push({ inicio: d - WINDOW_S, fin: d })
  return out
}

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
    const maxMuestras = Math.min(canal.length, Math.round(buffer.sampleRate * MAX_S))
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
