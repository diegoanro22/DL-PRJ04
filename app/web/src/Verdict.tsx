import { Info, Warning } from '@phosphor-icons/react'
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { Analisis } from './api'
import { SELLO, formatoProb } from './sello'

function Rosetta() {
  // Roseta de guilloche: 40 trazos cerrados desfasados, 24 lóbulos cada uno.
  const trazos: string[] = []
  for (let k = 0; k < 40; k++) {
    let d = ''
    for (let i = 0; i <= 720; i++) {
      const t = (i / 720) * Math.PI * 2
      const rho = 62 + 16 * Math.sin(24 * t + k * 0.157) + 7 * Math.sin(8 * t - k * 0.21)
      d += `${i ? 'L' : 'M'}${(rho * Math.cos(t)).toFixed(2)},${(rho * Math.sin(t)).toFixed(2)}`
    }
    trazos.push(d)
  }
  return (
    <svg className="rosette" viewBox="-100 -100 200 200" aria-hidden="true">
      {trazos.map((d, i) => <path key={i} d={d} />)}
      <circle r="92" />
      <circle r="88" />
      <circle r="30" />
    </svg>
  )
}

/** Marco de guilloche alrededor del sello, trazado al tamaño real de la caja. */
function Sello({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const [caja, setCaja] = useState({ w: 0, h: 0 })
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const medir = () => setCaja({ w: el.offsetWidth, h: el.offsetHeight })
    medir()
    const ro = new ResizeObserver(medir)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const { w, h } = caja
  const banda = 9
  const lineas: string[] = []
  if (w && h) {
    // recorre el perímetro a una distancia fija del borde y ondula hacia adentro y afuera
    const per = 2 * (w + h)
    const punto = (s: number): [number, number, number, number] => {
      const x0 = banda / 2, y0 = banda / 2, W = w - banda, H = h - banda
      s = ((s % per) + per) % per * ((2 * (W + H)) / per)
      if (s < W) return [x0 + s, y0, 0, -1]
      if ((s -= W) < H) return [x0 + W, y0 + s, 1, 0]
      if ((s -= H) < W) return [x0 + W - s, y0 + H, 0, 1]
      s -= W
      return [x0, y0 + H - s, -1, 0]
    }
    const ondas = Math.round(per / 14)
    for (let l = 0; l < 3; l++) {
      let d = ''
      for (let i = 0; i <= 900; i++) {
        const s = (i / 900) * per
        const [x, y, nx, ny] = punto(s)
        const a = (banda / 2 - 1) * Math.sin((2 * Math.PI * ondas * i) / 900 + (l * 2 * Math.PI) / 3)
        d += `${i ? 'L' : 'M'}${(x + nx * a).toFixed(1)},${(y + ny * a).toFixed(1)}`
      }
      lineas.push(d + 'Z')
    }
  }
  return (
    <div className="stamp" ref={ref}>
      <svg className="stamp__frame" width={w} height={h} aria-hidden="true">
        {lineas.map((d, i) => <path key={i} d={d} />)}
        <rect x={banda + 2} y={banda + 2} width={Math.max(0, w - 2 * banda - 4)} height={Math.max(0, h - 2 * banda - 4)} />
      </svg>
      {children}
    </div>
  )
}

interface Props {
  resultado: Analisis | null
  examinando: boolean
  clave: number
}

export function Verdict({ resultado, examinando, clave }: Props) {
  if (!resultado) {
    return (
      <section className="verdict verdict--vacio" aria-live="polite">
        <Rosetta />
        <p>{examinando ? 'Examinando…' : 'Aquí cae el sello con lo que hay que hacer.'}</p>
      </section>
    )
  }

  const { palabra, Icono } = SELLO[resultado.accion]
  const { t1, t2 } = resultado.umbrales
  const pos = (v: number) => `${Math.min(100, Math.max(0, v * 100))}%`

  return (
    <section className="verdict" data-accion={resultado.accion} data-examinando={examinando || undefined}
      aria-live="polite" aria-busy={examinando}>
      <Sello key={clave}>
        <Icono weight="bold" aria-hidden="true" />
        <span>{palabra}</span>
      </Sello>
      <p className="verdict__accion">{resultado.accion_texto}</p>

      <dl className="verdict__prob">
        {resultado.p_sintetica !== null ? (
          <>
            <dt>Probabilidad de voz sintética</dt>
            <dd className="prob__valor">{formatoProb(resultado.p_sintetica)}</dd>
            <dd className="prob__nota">
              calibrada para una prevalencia de {formatoProb(resultado.prevalencia_supuesta)} de ataques
            </dd>
          </>
        ) : (
          <>
            <dt>Score del modelo</dt>
            <dd className="prob__valor">{resultado.score.toFixed(4)}</dd>
            <dd className="prob__nota">Este modelo no está calibrado: el score no es una probabilidad.</dd>
          </>
        )}
      </dl>

      <figure className="scale">
        <figcaption>Score crudo {resultado.score.toFixed(4)} en la escala de umbrales</figcaption>
        <div className="scale__track" role="img"
          aria-label={`Score ${resultado.score.toFixed(3)}; verificar desde ${t1}, transferir desde ${t2}`}>
          <span className="zone zone--continuar" style={{ left: 0, width: pos(t1) }} />
          <span className="zone zone--verificar" style={{ left: pos(t1), width: `calc(${pos(t2)} - ${pos(t1)})` }} />
          <span className="zone zone--analista" style={{ left: pos(t2), right: 0 }} />
          <span className="scale__mark" style={{ left: pos(t1) }}><b>t1 {t1}</b></span>
          <span className="scale__mark" style={{ left: pos(t2) }}><b>t2 {t2}</b></span>
          <span className="scale__needle" style={{ left: pos(resultado.score) }} />
        </div>
        <div className="scale__ends" aria-hidden="true"><span>0 real</span><span>1 sintética</span></div>
      </figure>

      {resultado.umbrales_provisionales && (
        <p className="note note--provisional">
          <Info weight="bold" aria-hidden="true" />
          Umbrales provisionales: todavía no salen del modelo de valor.
        </p>
      )}
      {resultado.avisos.map((a) => (
        <p key={a} className="note note--aviso">
          <Warning weight="bold" aria-hidden="true" />
          {a}
        </p>
      ))}

      <dl className="verdict__meta">
        <div><dt>Modelo</dt><dd>{resultado.modelo}</dd></div>
        <div><dt>Audio</dt><dd>{resultado.duracion_s.toFixed(1)} s</dd></div>
        <div><dt>Inferencia</dt><dd>{Math.round(resultado.latencia_ms)} ms</dd></div>
      </dl>
    </section>
  )
}
