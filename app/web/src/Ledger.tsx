import type { Analisis, MetricasModelo } from './api'
import { SELLO, formatoProb } from './sello'

export interface Entrada {
  id: number
  hora: string
  fuente: string
  resultado: Analisis
}

interface Props {
  entradas: Entrada[]
  metricas: MetricasModelo | null
}

export function Ledger({ entradas, metricas }: Props) {
  return (
    <section className="ledger" aria-labelledby="ledger-titulo">
      <header className="ledger__head">
        <h2 id="ledger-titulo">Registro de esta sesión</h2>
        {metricas && (
          <p className="ledger__monitor">
            <span>{metricas.llamadas} análisis en el servicio</span>
            <span>
              alertas recientes {formatoProb(metricas.tasa_alertas_reciente)}
              {metricas.tasa_alertas_esperada !== null && ` (esperado ${formatoProb(metricas.tasa_alertas_esperada)})`}
            </span>
            <span>p95 {Math.round(metricas.latencia_ms.p95)} ms</span>
            {metricas.alarma_deriva && <strong className="ledger__alarma">Alarma de deriva: la tasa de alertas se salió de lo esperado</strong>}
          </p>
        )}
      </header>

      {entradas.length === 0 ? (
        <p className="ledger__vacio">
          Cada análisis queda sellado aquí. No se guarda el audio, solo el resultado y solo en esta pestaña.
        </p>
      ) : (
        <div className="ledger__scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Hora</th>
                <th scope="col">Audio</th>
                <th scope="col">Acción</th>
                <th scope="col" className="num">P(sintética)</th>
                <th scope="col" className="num col-opcional">Score</th>
                <th scope="col" className="num col-opcional">Inferencia</th>
                <th scope="col" className="col-opcional">Modelo</th>
              </tr>
            </thead>
            <tbody>
              {entradas.map(({ id, hora, fuente, resultado: r }) => {
                const { palabra, Icono } = SELLO[r.accion]
                return (
                  <tr key={id} data-accion={r.accion}>
                    <td className="num">{hora}</td>
                    <td className="ledger__fuente">{fuente}</td>
                    <td>
                      <span className="mini-stamp" data-accion={r.accion}>
                        <Icono weight="bold" aria-hidden="true" />
                        {palabra}
                      </span>
                    </td>
                    <td className="num">{r.p_sintetica === null ? 'sin calibrar' : formatoProb(r.p_sintetica)}</td>
                    <td className="num col-opcional">{r.score.toFixed(4)}</td>
                    <td className="num col-opcional">{Math.round(r.latencia_ms)} ms</td>
                    <td className="col-opcional">{r.modelo}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
