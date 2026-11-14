import { useCallback, useEffect, useRef, useState } from 'react'
import { BIN_S, MAX_RECORD_S, normalizar } from './audio'

export type EstadoGrabacion = 'inactivo' | 'pidiendo' | 'grabando'

interface Resultado {
  blob: Blob
  nombre: string
}

/**
 * Graba del micrófono sin cancelación de eco, supresión de ruido ni control de ganancia:
 * ese procesamiento cambia el espectro y el detector vería otra voz.
 * Mientras graba expone la envolvente en vivo para dibujar la banda.
 */
export function useRecorder(onDone: (r: Resultado) => void) {
  const [estado, setEstado] = useState<EstadoGrabacion>('inactivo')
  const [segundos, setSegundos] = useState(0)
  const [envolvente, setEnvolvente] = useState<number[]>([])
  const [error, setError] = useState<string | null>(null)

  const recorder = useRef<MediaRecorder | null>(null)
  const stream = useRef<MediaStream | null>(null)
  const ctx = useRef<AudioContext | null>(null)
  const raf = useRef<number>(0)
  const timeout = useRef<number>(0)
  const doneRef = useRef(onDone)
  useEffect(() => {
    doneRef.current = onDone
  }, [onDone])

  const limpiar = useCallback(() => {
    cancelAnimationFrame(raf.current)
    clearTimeout(timeout.current)
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
    void ctx.current?.close()
    ctx.current = null
  }, [])

  useEffect(() => limpiar, [limpiar])

  const detener = useCallback(() => {
    if (recorder.current?.state === 'recording') recorder.current.stop()
  }, [])

  const iniciar = useCallback(async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('Este navegador no permite grabar. Suba un archivo de audio.')
      return
    }
    setEstado('pidiendo')
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      })
    } catch {
      setEstado('inactivo')
      setError('No se pudo usar el micrófono. Revise el permiso del navegador.')
      return
    }

    const tipo = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'].find((t) =>
      MediaRecorder.isTypeSupported(t),
    )
    const rec = new MediaRecorder(stream.current, tipo ? { mimeType: tipo } : undefined)
    const partes: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && partes.push(e.data)
    rec.onstop = () => {
      limpiar()
      setEstado('inactivo')
      const mime = rec.mimeType || 'audio/webm'
      const ext = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'm4a' : 'webm'
      const blob = new Blob(partes, { type: mime })
      if (blob.size) doneRef.current({ blob, nombre: `microfono.${ext}` })
    }
    recorder.current = rec

    // Envolvente en vivo con un AnalyserNode.
    ctx.current = new AudioContext()
    const fuente = ctx.current.createMediaStreamSource(stream.current)
    const analyser = ctx.current.createAnalyser()
    analyser.fftSize = 2048
    fuente.connect(analyser)
    const buf = new Float32Array(analyser.fftSize)
    const crudos: number[] = []
    const t0 = performance.now()
    let ultimoBin = 0
    const paso = () => {
      const t = (performance.now() - t0) / 1000
      analyser.getFloatTimeDomainData(buf)
      let suma = 0
      for (let i = 0; i < buf.length; i++) suma += buf[i] * buf[i]
      const rms = Math.sqrt(suma / buf.length)
      const bins = Math.floor(t / BIN_S)
      while (ultimoBin < bins) {
        crudos.push(rms)
        ultimoBin++
      }
      setSegundos(t)
      setEnvolvente(normalizar(crudos))
      raf.current = requestAnimationFrame(paso)
    }

    setEnvolvente([])
    setSegundos(0)
    rec.start()
    setEstado('grabando')
    raf.current = requestAnimationFrame(paso)
    timeout.current = window.setTimeout(detener, MAX_RECORD_S * 1000)
  }, [detener, limpiar])

  return { estado, segundos, envolvente, error, iniciar, detener }
}
