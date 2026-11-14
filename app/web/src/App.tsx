import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowClockwise, Microphone, Stop, UploadSimple, WaveformSlash } from '@phosphor-icons/react'
import { analizar, getMetricas, getModelos, getSalud, type Analisis, type Metricas, type ModeloInfo, type Salud } from './api'
import { decodificar, MAX_RECORD_S, WINDOW_S } from './audio'
import { useRecorder } from './useRecorder'
import { SecurityBand } from './SecurityBand'
import { Verdict } from './Verdict'
import { Ledger, type Entrada } from './Ledger'

interface AudioCargado {
  blob: Blob
  nombre: string
  envolvente: number[]
  duracion: number
}

const ESCANEO_MIN_MS = 700 // el barrido UV se ve completo al menos una vez

function useReducirMovimiento() {
  const [reducir, setReducir] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const fn = () => setReducir(mq.matches)
    mq.addEventListener('change', fn)
    return () => mq.removeEventListener('change', fn)
  }, [])
  return reducir
}

export default function App() {
  const [salud, setSalud] = useState<Salud | null>(null)
  const [modelos, setModelos] = useState<ModeloInfo[]>([])
  const [modelo, setModelo] = useState<string | null>(null)
  const [prevalencia, setPrevalencia] = useState('0.5')
  const [audio, setAudio] = useState<AudioCargado | null>(null)
  const [resultado, setResultado] = useState<Analisis | null>(null)
  const [examinando, setExaminando] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [entradas, setEntradas] = useState<Entrada[]>([])
  const [metricas, setMetricas] = useState<Metricas | null>(null)
  const [clave, setClave] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const ocupado = useRef(false)
  const reducir = useReducirMovimiento()

  useEffect(() => {
    getSalud()
      .then((s) => {
        setSalud(s)
        setModelo((m) => m ?? s.modelo_por_defecto)
      })
      .catch((e: Error) => setError(e.message))
    getModelos().then(setModelos).catch(() => {})
    getMetricas().then(setMetricas).catch(() => {})
  }, [])

  const prev = Math.min(50, Math.max(0.01, Number(prevalencia) || 0.5)) / 100

  const ejecutar = useCallback(
    async (a: AudioCargado) => {
      if (ocupado.current) return
      ocupado.current = true
      setExaminando(true)
      setError(null)
      try {
        const [r] = await Promise.all([
          analizar(a.blob, a.nombre, modelo, prev),
          new Promise((ok) => setTimeout(ok, reducir ? 0 : ESCANEO_MIN_MS)),
        ])
        setResultado(r)
        setClave((k) => k + 1)
        setEntradas((es) => [
          {
            id: Date.now(),
            hora: new Date().toLocaleTimeString('es-GT', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }),
            fuente: a.nombre,
            resultado: r,
          },
          ...es,
        ].slice(0, 50))
        getMetricas().then(setMetricas).catch(() => {})
      } catch (e) {
        setError((e as Error).message)
      } finally {
        ocupado.current = false
        setExaminando(false)
      }
    },
    [modelo, prev, reducir],
  )

  const cargar = useCallback(
    async (blob: Blob, nombre: string) => {
      // Un solo análisis a la vez: dos respuestas en vuelo podrían llegar en desorden y dejar
      // en pantalla el veredicto del audio anterior. Mientras graba, la banda y el botón de
      // subir ya ignoran archivos nuevos.
      if (ocupado.current) return
      ocupado.current = true
      setError(null)
      let a: AudioCargado
      try {
        const env = await decodificar(blob)
        a = { blob, nombre, envolvente: env?.valores ?? [], duracion: env?.duracion ?? WINDOW_S }
        setAudio(a)
      } finally {
        ocupado.current = false
      }
      await ejecutar(a)
    },
    [ejecutar],
  )

  const rec = useRecorder(({ blob, nombre }) => void cargar(blob, nombre))
  const grabando = rec.estado === 'grabando'

  const modoBanda = grabando ? 'grabando' : examinando ? 'examinando' : audio ? 'listo' : 'vacio'
  const envolvente = grabando ? rec.envolvente : audio?.envolvente ?? []
  const duracion = grabando ? rec.segundos : audio?.duracion ?? 0
  const sinModelos = salud?.estado === 'sin_modelos'
  const infoModelo = modelos.find((m) => m.nombre === (modelo ?? salud?.modelo_por_defecto))
  const mensajeError = rec.error ?? error

  return (
    <div className="page">
      <header className="topbar">
        <h1>Verificador de voz</h1>
        <div className="topbar__controls">
          <label className="field">
            <span>Modelo</span>
            <select value={modelo ?? ''} onChange={(e) => setModelo(e.target.value)} disabled={!modelos.length}>
              {modelos.map((m) => (
                <option key={m.nombre} value={m.nombre}>
                  {m.nombre}{m.calibrado ? '' : ' (sin calibrar)'}
                </option>
              ))}
            </select>
          </label>
          <label className="field field--prev">
            <span>Prevalencia de ataques</span>
            <span className="field__unit">
              <input type="number" inputMode="decimal" min="0.01" max="50" step="0.1" value={prevalencia}
                onChange={(e) => setPrevalencia(e.target.value)} />
              %
            </span>
          </label>
          <p className="status" data-estado={salud ? salud.estado : error ? 'error' : 'cargando'}>
            {salud ? (sinModelos ? 'Sin modelos cargados' : 'Servicio en línea') : error ? 'Sin conexión' : 'Conectando…'}
          </p>
        </div>
      </header>

      <main className="desk">
        <section className="examine" aria-label="Audio de la llamada">
          <SecurityBand envolvente={envolvente} duracion={duracion} modo={modoBanda}
            onArchivo={(f) => void cargar(f, f.name)} reducirMovimiento={reducir}
            aceptaArchivos={!grabando && !examinando && !sinModelos} />

          <div className="levers">
            {grabando ? (
              <button className="btn btn--primary btn--rec" onClick={rec.detener}>
                <Stop weight="fill" aria-hidden="true" />
                Detener
                <span className="btn__timer" aria-live="off">
                  {rec.segundos.toFixed(1)} / {MAX_RECORD_S} s
                </span>
              </button>
            ) : (
              <button className="btn btn--primary" onClick={() => void rec.iniciar()}
                disabled={rec.estado === 'pidiendo' || examinando || sinModelos}>
                <Microphone weight="bold" aria-hidden="true" />
                {rec.estado === 'pidiendo' ? 'Esperando permiso…' : 'Grabar'}
              </button>
            )}
            <button className="btn" onClick={() => input.current?.click()} disabled={grabando || examinando || sinModelos}>
              <UploadSimple weight="bold" aria-hidden="true" />
              Subir audio
            </button>
            <input ref={input} type="file" accept="audio/*,.wav,.flac,.ogg,.webm,.mp3" hidden
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void cargar(f, f.name)
                e.target.value = ''
              }} />
            {audio && !grabando && (
              <button className="btn btn--quiet" onClick={() => void ejecutar(audio)} disabled={examinando}>
                <ArrowClockwise weight="bold" aria-hidden="true" />
                Volver a analizar
              </button>
            )}
          </div>

          {mensajeError && (
            <p className="alert" role="alert">
              <WaveformSlash weight="bold" aria-hidden="true" />
              {mensajeError}
            </p>
          )}
          {sinModelos && (
            <p className="alert" role="alert">
              <WaveformSlash weight="bold" aria-hidden="true" />
              La API no tiene modelos. Empaquete uno con python -m app.package_model.
            </p>
          )}
        </section>

        <Verdict resultado={resultado} examinando={examinando} clave={clave} />
      </main>

      <Ledger entradas={entradas} metricas={(infoModelo && metricas?.modelos[infoModelo.nombre]) || null} />

      <footer className="foot">
        <p>
          Detector entrenado sobre ASVspoof 2019 LA. Analiza solo los primeros 4 s y no guarda el audio.
          Es un apoyo para el agente: la decisión final sigue los procedimientos del banco.
        </p>
      </footer>
    </div>
  )
}
