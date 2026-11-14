import { PhoneTransfer, ShieldCheck, ShieldWarning } from '@phosphor-icons/react'
import type { Accion } from './api'

export const SELLO: Record<Accion, { palabra: string; Icono: typeof ShieldCheck }> = {
  continuar: { palabra: 'Continuar', Icono: ShieldCheck },
  verificar: { palabra: 'Verificar', Icono: ShieldWarning },
  analista: { palabra: 'Transferir', Icono: PhoneTransfer },
}

export function formatoProb(p: number): string {
  const pct = p * 100
  if (pct < 0.01) return '< 0.01 %'
  if (pct >= 99.995) return '> 99.99 %'
  if (pct < 1 || pct >= 99) return `${pct.toFixed(2)} %`
  if (pct < 10 || pct >= 90) return `${pct.toFixed(1)} %`
  return `${pct.toFixed(0)} %`
}
