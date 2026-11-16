import { useMemo, useState, type DragEvent } from 'react'
import { BIN_S, WINDOW_S } from './audio'

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
}

/**
 * La banda de seguridad: un guilloche cuyas líneas se abren con la energía real del audio.
 * La ventana que ve el modelo (0 a 4 s) va en tinta llena; lo que el modelo no ve queda tenue,
 * y si el audio dura menos de 4 s, la parte que el modelo repite se marca como repetida.
 */
export function SecurityBand({ envolvente, duracion, modo, onArchivo, reducirMovimiento, aceptaArchivos }: Props) {
  const [arrastrando, setArrastrando] = useState(false)
  const vacio = modo === 'vacio'
  const span = Math.max(WINDOW_S, Math.min(duracion || 0, 10))
  const xVentana = (WINDOW_S / span) * W
  const xDuracion = vacio ? W : Math.min(W, (duracion / span) * W)
  const repetido = !vacio && modo !== 'grabando' && duracion > 0 && duracion < WINDOW_S

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
  for (let t = 0; t <= span + 1e-6; t += 0.5) ticks.push(t)

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
            <clipPath id="clip-visto"><rect x="0" y="0" width={Math.min(xDuracion, xVentana)} height={H} /></clipPath>
            <clipPath id="clip-repetido"><rect x={xDuracion} y="0" width={Math.max(0, xVentana - xDuracion)} height={H} /></clipPath>
            <clipPath id="clip-fuera"><rect x={xVentana} y="0" width={W - xVentana} height={H} /></clipPath>
            <clipPath id="clip-uv">
              <rect x="-140" y="0" width="140" height={H}>
                {!reducirMovimiento && modo === 'examinando' && (
                  <animate attributeName="x" from="-140" to={xVentana} dur="1.15s" repeatCount="indefinite"
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

          <line x1={xVentana} x2={xVentana} y1="0" y2={H} className="band__window" vectorEffect="non-scaling-stroke" />
        </svg>

        {vacio && (
          <p className="band__empty">
            <span>Grabe o suelte aquí los primeros segundos de la llamada</span>
          </p>
        )}
      </div>

      <div className="band__ruler" aria-hidden="true">
        <div className="ruler__micro">
          {Array.from({ length: 24 }, (_, i) => <span key={i}>{i % 2 ? 'CUATRO SEGUNDOS' : 'SIGNET'}</span>)}
        </div>
        {ticks.map((t) => (
          <span key={t} className={Number.isInteger(t) ? 'tick tick--mayor' : 'tick'}
            style={{ left: `${(t / span) * 100}%` }}>
            {Number.isInteger(t) && t !== WINDOW_S && <b>{t}</b>}
          </span>
        ))}
        <span className={xVentana / W > 0.6 ? 'ruler__window ruler__window--fin' : 'ruler__window'} style={{ left: `${(xVentana / W) * 100}%` }}>
          4 s: lo que analiza el modelo
        </span>
      </div>

      <figcaption className="band__caption">
        {modo === 'grabando' && 'Grabando. Con 4 s ya alcanza; se detiene sola a los 8 s.'}
        {modo === 'examinando' && 'Examinando la ventana de 4 s…'}
        {modo === 'listo' && repetido && `El audio dura ${duracion.toFixed(1)} s: el modelo lo repite hasta completar 4 s (tramo tenue).`}
        {modo === 'listo' && !repetido && duracion > WINDOW_S && `El audio dura ${duracion.toFixed(1)} s: solo se analizan los primeros 4 s.`}
        {modo === 'listo' && !repetido && duracion > 0 && duracion <= WINDOW_S && duracion >= WINDOW_S - 0.05 && 'Ventana completa de 4 s.'}
        {vacio && 'También acepta WAV, FLAC, OGG, WebM o MP3.'}
      </figcaption>
    </figure>
  )
}
