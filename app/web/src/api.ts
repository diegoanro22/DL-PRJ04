// Cliente de la API (mismo origen; en desarrollo Vite hace de proxy).

export type Accion = 'continuar' | 'verificar' | 'analista'

export interface Analisis {
  modelo: string
  score: number
  p_sintetica: number | null
  prevalencia_supuesta: number
  accion: Accion
  accion_texto: string
  umbrales: { t1: number; t2: number }
  umbrales_provisionales: boolean
  duracion_s: number
  avisos: string[]
  latencia_ms: number
}

export interface ModeloInfo {
  nombre: string
  umbrales: { t1: number; t2: number }
  umbrales_provisionales: boolean
  calibrado: boolean
}

export interface Salud {
  estado: 'ok' | 'sin_modelos'
  modelos: string[]
  modelo_por_defecto: string | null
}

export interface MetricasModelo {
  llamadas: number
  acciones: Record<Accion, number>
  recientes: number
  tasa_alertas_reciente: number
  tasa_alertas_esperada: number | null
  alarma_deriva: boolean
  latencia_ms: { p50: number; p95: number }
  histograma_scores: number[]
}

export interface Metricas {
  desde: string
  modelos: Record<string, MetricasModelo>
  errores: Record<string, number>
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError(0, 'No hay conexión con el servicio. Revise la red o que la API esté levantada.')
  }
  if (!res.ok) {
    let detail = `Error ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } catch {
      /* respuesta sin JSON */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

export const getSalud = () => request<Salud>('/health')
export const getModelos = () => request<ModeloInfo[]>('/models')
export const getMetricas = () => request<Metricas>('/metrics')

export function analizar(audio: Blob, nombre: string, modelo: string | null, prevalencia: number) {
  const form = new FormData()
  form.append('audio', audio, nombre)
  const params = new URLSearchParams({ prevalencia: String(prevalencia) })
  if (modelo) params.set('modelo', modelo)
  return request<Analisis>(`/analyze?${params}`, { method: 'POST', body: form })
}
