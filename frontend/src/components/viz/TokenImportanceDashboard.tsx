import { useEffect } from 'react'
import { Scales } from '@phosphor-icons/react'
import { useTokenImportance } from '../../api/hooks'
import ApiLoadingState from '../education/ApiLoadingState'

interface TokenImportanceDashboardProps {
  tokens?: string[]
  d_model?: number
}

const FALLBACK = {
  tokens: ['O', 'gato', 'pulou', 'no', 'jardim'],
  importancia_recebida: [0.12, 0.35, 0.28, 0.08, 0.17],
  importancia_dada: [0.15, 0.22, 0.30, 0.10, 0.23],
  importancia_combinada: [0.14, 0.29, 0.29, 0.09, 0.20],
  token_mais_importante: 'gato',
  explicacao: 'Dados de exemplo - inicie o backend para resultados reais.',
}

function classifyRole(recebida: number, dada: number, maxR: number, maxD: number): { label: string; color: string } {
  const rNorm = maxR > 0 ? recebida / maxR : 0
  const dNorm = maxD > 0 ? dada / maxD : 0
  if (rNorm > 0.6 && dNorm > 0.6) return { label: 'Hub', color: 'bg-violet-100 text-violet-700 border-violet-200' }
  if (dNorm > 0.6) return { label: 'Fonte', color: 'bg-blue-100 text-blue-700 border-blue-200' }
  if (rNorm > 0.6) return { label: 'Sumidouro', color: 'bg-amber-100 text-amber-700 border-amber-200' }
  return { label: 'Neutro', color: 'bg-gray-100 text-gray-600 border-gray-200' }
}

export default function TokenImportanceDashboard({ tokens = ['O', 'gato', 'pulou', 'no', 'jardim'], d_model = 64 }: TokenImportanceDashboardProps) {
  const importance = useTokenImportance()

  useEffect(() => {
    if (tokens.length > 0) {
      importance.execute({ tokens, d_model })
    }
  }, [tokens.join(','), d_model])

  const data = importance.data ?? FALLBACK
  const maxRecebida = Math.max(...data.importancia_recebida)
  const maxDada = Math.max(...data.importancia_dada)
  const maxCombinada = Math.max(...data.importancia_combinada)

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Scales size={16} weight="duotone" className="text-violet-500" />
        <h4 className="text-sm font-semibold text-gray-800">Dashboard de Importancia dos Tokens</h4>
      </div>
      <p className="text-xs text-gray-500">
        Barras divergentes mostram atencao recebida (esquerda) vs dada (direita). Tokens sao classificados por papel na rede de atencao.
      </p>

      <ApiLoadingState
        loading={importance.loading}
        error={importance.error}
        compact
        fallback={
          <div className="text-xs text-gray-400 text-center py-4">
            Dados de exemplo - inicie o backend para resultados reais.
          </div>
        }
      >
        <div className="space-y-4">
          {/* Legend */}
          <div className="flex flex-wrap gap-2 text-[10px]">
            {[
              { label: 'Hub', desc: 'doa e recebe muita atencao', color: 'bg-violet-100 text-violet-700 border-violet-200' },
              { label: 'Fonte', desc: 'doa atencao', color: 'bg-blue-100 text-blue-700 border-blue-200' },
              { label: 'Sumidouro', desc: 'recebe atencao', color: 'bg-amber-100 text-amber-700 border-amber-200' },
              { label: 'Neutro', desc: 'baixa importancia', color: 'bg-gray-100 text-gray-600 border-gray-200' },
            ].map((r) => (
              <span key={r.label} className={`px-1.5 py-0.5 rounded-sm border ${r.color}`}>
                {r.label}: {r.desc}
              </span>
            ))}
          </div>

          {/* Divergent bars */}
          <div className="space-y-2">
            <div className="grid grid-cols-[1fr_auto_1fr] gap-1 items-center text-[10px] text-gray-400 font-mono mb-1">
              <span className="text-right">Atencao Recebida</span>
              <span className="w-20 text-center">Token</span>
              <span>Atencao Dada</span>
            </div>
            {data.tokens.map((token, i) => {
              const recebida = data.importancia_recebida[i]
              const dada = data.importancia_dada[i]
              const role = classifyRole(recebida, dada, maxRecebida, maxDada)
              const recWidth = maxRecebida > 0 ? (recebida / maxRecebida) * 100 : 0
              const dadaWidth = maxDada > 0 ? (dada / maxDada) * 100 : 0

              return (
                <div key={i} className="grid grid-cols-[1fr_auto_1fr] gap-1 items-center group">
                  {/* Left bar - received */}
                  <div className="flex items-center justify-end gap-1">
                    <span className="text-[10px] font-mono text-gray-400 group-hover:text-gray-600 transition-colors">
                      {recebida.toFixed(3)}
                    </span>
                    <div className="w-full max-w-[200px] bg-gray-100 rounded-full h-3 overflow-hidden flex justify-end">
                      <div
                        className="bg-amber-400 h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.max(3, recWidth)}%` }}
                      />
                    </div>
                  </div>

                  {/* Token label */}
                  <div className="w-20 text-center">
                    <span className={`px-1.5 py-0.5 rounded-sm border text-[11px] font-mono font-medium ${role.color}`}>
                      {token}
                    </span>
                  </div>

                  {/* Right bar - given */}
                  <div className="flex items-center gap-1">
                    <div className="w-full max-w-[200px] bg-gray-100 rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-blue-400 h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.max(3, dadaWidth)}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-gray-400 group-hover:text-gray-600 transition-colors">
                      {dada.toFixed(3)}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Combined importance */}
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">Importancia Combinada</p>
            <div className="space-y-1.5">
              {data.tokens.map((token, i) => {
                const combined = data.importancia_combinada[i]
                const width = maxCombinada > 0 ? (combined / maxCombinada) * 100 : 0
                const isTop = token === data.token_mais_importante

                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className={`text-[11px] font-mono w-16 text-right ${isTop ? 'text-violet-600 font-semibold' : 'text-gray-500'}`}>
                      {token}
                    </span>
                    <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${isTop ? 'bg-violet-500' : 'bg-gray-300'}`}
                        style={{ width: `${Math.max(3, width)}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-gray-400 w-10 text-right">
                      {combined.toFixed(3)}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Explanation */}
          <p className="text-xs text-gray-500 leading-relaxed">
            {data.explicacao}
          </p>
        </div>
      </ApiLoadingState>
    </div>
  )
}
