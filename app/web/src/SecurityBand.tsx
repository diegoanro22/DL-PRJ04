import { useMemo, useState, type DragEvent } from 'react'
import type { Analisis } from './api'
import { BIN_S, MAX_S, WINDOW_S, ventanasDe } from './audio'

const W = 1000
const H = 200
const LINEAS = 13
const PASO_X = 4

interface Props {
  /** RMS normalizado por bin de BIN_S segundos; vacío si no hay audio */
  envolvente: number[]
  duracion: number
  modo: 'vacio' | 'grabando' | 'listo' | 'examinando'
  onArchivo: (f: File) => void
  reducirMovimiento: boolean
  /** false mientras graba o analiza: el audio soltado se ignora */
  aceptaArchivos: boolean
  /** ventanas del último análisis de este mismo audio (si ya hay resultado) */
  ventanas?: Analisis['ventanas']
  /** umbral de transferir, para marcar las ventanas que lo pasan */
  t2?: number
}

/**
 * La banda de seguridad: un guilloche cuyas líneas se abren con la energía real del audio.
 * El modelo lee el audio en ventanas de 4 s (hasta 20 s); debajo de la regla va el score de
 * cada ventana. Si el audio dura menos de 4 s, la parte que el modelo repite se marca tenue.
 */
export function SecurityBand({ envolvente, duracion, modo, onArchivo, reducirMovimiento, aceptaArchivos, ventanas, t2 }: Props) {
  const [arrastrando, setArrastrando] = useState(false)
  const vacio = modo === 'vacio'
  const span = Math.max(WINDOW_S, Math.min(duracion || 0, MAX_S))
  const xVentana = (WINDOW_S / span) * W
  const xDuracion = vacio ? W : Math.min(W, (duracion / span) * W)
  const repetido = !vacio && modo !== 'grabando' && duracion > 0 && duracion < WINDOW_S
  // Hasta donde lee el modelo: todo el audio, o la ventana completa si se repite.
  const xLeido = repetido ? xVentana : xDuracion
  const tramos = vacio || modo === 'grabando' ? [] : ventanasDe(duracion)
  // Celdas sin solapar: la última ventana se alinea al final y pisa a la anterior.
  const celdas = (ventanas?.length
    ? ventanas.map((v) => ({ inicio: v.inicio_s, fin: v.fin_s, score: v.score as number | null, usada: v.usada }))
    : tramos.map((v) => ({ ...v, score: null as number | null, usada: true }))
  ).map((c, i, arr) => ({ ...c, desde: i ? Math.max(c.inicio, arr[i - 1].fin) : c.inicio }))
  const usadas = celdas.filter((c) => c.usada).length

  const paths = useMemo(() => {
    const energia = (t: number) => {
      if (vacio) return 0.16
      let tt = t
      if (t > duracion) {
        if (!repetido || t > WINDOW_S || duracion <= 0) return 0
        tt = t % duracion // el modelo repite el audio hasta completar 4 s
      }
      const i = Math.min(envolvente.length - 1, Math.floor(tt / BIN_S))
      if (i < 0) return 0
      // suavizado corto para que el trazo no tiemble
      const a = envolvente[Math.max(0, i - 1)] ?? 0
      const b = envolvente[i] ?? 0
      const c = envolvente[Math.min(envolvente.length - 1, i + 1)] ?? 0
      return (a + 2 * b + c) / 4
    }
    const out: string[] = []
    for (let l = 0; l < LINEAS; l++) {
      let d = ''
      for (let x = 0; x <= W; x += PASO_X) {
        const t = (x / W) * span
        const amp = 5 + energia(t) * 78
        const y =
          H / 2 +
          amp *
            (0.6 * Math.sin((2 * Math.PI * x) / 44 + l * 0.48) +
              0.4 * Math.sin((2 * Math.PI * x) / 131 - l * 0.29))
        d += `${x === 0 ? 'M' : 'L'}${x},${y.toFixed(1)}`
      }
      out.push(d)
    }
    return out
  }, [envolvente, duracion, span, vacio, repetido])

  const grupo = (cls: string) => (
    <g className={cls}>
      {paths.map((d, i) => (
        <path key={i} d={d} vectorEffect="non-scaling-stroke" />
      ))}
    </g>
  )

  const ticks = []
  const paso = span > 10 ? 1 : 0.5
  for (let t = 0; t <= span + 1e-6; t += paso) ticks.push(t)

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setArrastrando(false)
    const f = e.dataTransfer.files[0]
    if (f && aceptaArchivos) onArchivo(f)
  }

  return (
    <figure
      className="band"
      data-modo={modo}
      data-arrastrando={arrastrando || undefined}
      onDragOver={(e) => {
        e.preventDefault() // siempre: si no, el navegador abre el archivo
        e.dataTransfer.dropEffect = aceptaArchivos ? 'copy' : 'none'
        setArrastrando(aceptaArchivos)
      }}
      onDragLeave={() => setArrastrando(false)}
      onDrop={onDrop}
    >
      <div className="band__plate">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
          aria-label={vacio ? 'Sin audio' : `Forma del audio, ${duracion.toFixed(1)} segundos`}>
          <defs>
            <clipPath id="clip-visto"><rect x="0" y="0" width={xDuracion} height={H} /></clipPath>
            <clipPath id="clip-repetido"><rect x={xDuracion} y="0" width={Math.max(0, xVentana - xDuracion)} height={H} /></clipPath>
            <clipPath id="clip-fuera"><rect x={xLeido} y="0" width={W - xLeido} height={H} /></clipPath>
            <clipPath id="clip-uv">
              <rect x="-140" y="0" width="140" height={H}>
                {!reducirMovimiento && modo === 'examinando' && (
                  <animate attributeName="x" from="-140" to={xLeido} dur="1.15s" repeatCount="indefinite"
                    calcMode="spline" keySplines="0.45 0 0.55 1" keyTimes="0;1" />
                )}
              </rect>
            </clipPath>
          </defs>

          <g clipPath="url(#clip-visto)">{grupo('band__ink')}</g>
          {repetido && <g clipPath="url(#clip-repetido)">{grupo('band__ink band__ink--repetido')}</g>}
          <g clipPath="url(#clip-fuera)">{grupo('band__ink band__ink--fuera')}</g>

          {modo === 'examinando' && (
            reducirMovimiento
              ? <g clipPath="url(#clip-visto)">{grupo('band__uv')}</g>
              : <g clipPath="url(#clip-uv)">
                  <rect x="0" y="0" width={W} height={H} className="band__uv-glow" />
                  {grupo('band__uv')}
                </g>
          )}

          {celdas.slice(1).map((c) => {
            const x = (c.desde / span) * W
            return <line key={c.desde} x1={x} x2={x} y1="0" y2={H} className="band__window" vectorEffect="non-scaling-stroke" />
          })}
        </svg>

        {vacio && (
          <p className="band__empty">
            <span>Grabe o suelte aquí el audio de la llamada</span>
          </p>
        )}
      </div>

      <div className="band__ruler" aria-hidden="true">
        <div className="ruler__micro">
          {Array.from({ length: 24 }, (_, i) => <span key={i}>{i % 2 ? 'VENTANAS DE CUATRO SEGUNDOS' : 'SIGNET'}</span>)}
        </div>
        {ticks.map((t) => (
          <span key={t} className={Number.isInteger(t) ? 'tick tick--mayor' : 'tick'}
            style={{ left: `${(t / span) * 100}%` }}>
            {Number.isInteger(t) && (span <= 10 || t % 2 === 0) && <b>{t}</b>}
          </span>
        ))}
      </div>

      {celdas.length > 0 && (
        <ol className="band__ventanas" aria-label="Score de cada ventana de 4 s">
          {celdas.map((c, i) => (
            <li key={c.desde} className="ventana" data-usada={c.usada}
              title={c.score === null ? undefined : `${c.inicio.toFixed(1)}–${c.fin.toFixed(1)} s: score ${c.score.toFixed(4)}${c.usada ? '' : ' (silencio, no vota)'}`}
              data-alto={c.score !== null && t2 !== undefined && c.score >= t2 ? true : undefined}
              style={{ left: `${(c.desde / span) * 100}%`, width: `${((c.fin - c.desde) / span) * 100}%` }}>
              <span className="ventana__barra" style={{ width: `${(c.score ?? 0) * 100}%` }} />
              {/* La última ventana se recorta donde termina la anterior: si queda angosta, sin texto. */}
              {(c.fin - c.desde) / span >= 0.09 && (
                <span className="ventana__txt">
                  {c.score === null ? `V${i + 1}` : !c.usada ? 'silencio' : c.score < 0.01 ? '< 0.01' : c.score.toFixed(2)}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}

      <figcaption className="band__caption">
        {modo === 'grabando' && `Grabando. Con 8 s o más el modelo tiene varias ventanas; se detiene sola a los ${MAX_S} s.`}
        {modo === 'examinando' && (celdas.length > 1 ? `Examinando ${celdas.length} ventanas de 4 s…` : 'Examinando la ventana de 4 s…')}
        {modo === 'listo' && repetido && `El audio dura ${duracion.toFixed(1)} s: el modelo lo repite hasta completar 4 s (tramo tenue).`}
        {modo === 'listo' && !repetido && celdas.length > 1 && (ventanas?.length
          ? `${celdas.length} ventanas de 4 s; la decisión promedia ${usadas === celdas.length ? 'todas' : `las ${usadas} con voz (las de silencio no votan)`}.`
          : `${celdas.length} ventanas de 4 s.`)}
        {modo === 'listo' && !repetido && celdas.length === 1 && 'Una ventana de 4 s. Grabe 8 s o más para que la decisión promedie varias.'}
        {vacio && 'También acepta WAV, FLAC, OGG, WebM o MP3.'}
      </figcaption>
    </figure>
  )
}
