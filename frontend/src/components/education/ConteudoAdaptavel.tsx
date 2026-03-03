import { useOutletContext } from 'react-router-dom'
import type { LayoutContext } from '../layout/Layout'

interface ConteudoAdaptavelProps {
  iniciante: React.ReactNode
  avancado: React.ReactNode
}

export default function ConteudoAdaptavel({ iniciante, avancado }: ConteudoAdaptavelProps) {
  const { nivelConhecimento } = useOutletContext<LayoutContext>()
  return <>{nivelConhecimento === 'iniciante' ? iniciante : avancado}</>
}
