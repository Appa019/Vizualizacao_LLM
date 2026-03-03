import { useEffect } from 'react'
import { Shuffle } from '@phosphor-icons/react'
import { useSamplingDemo } from '../../api/hooks'
import ApiLoadingState from '../education/ApiLoadingState'

interface SamplingComparisonProps {
  prompt?: string[]
}

const STRATEGIES = [
  { key: 'greedy', label: 'Greedy', color: 'blue', desc: 'Sempre o mais provavel' },
  { key: 'temperatura_baixa', label: 'Temp. Baixa', color: 'green', desc: 'Conservador' },
  { key: 'temperatura_alta', label: 'Temp. Alta', color: 'red', desc: 'Criativo / caotico' },
  { key: 'top_k', label: 'Top-K', color: 'purple', desc: 'Amostra dos K melhores' },
  { key: 'top_p', label: 'Top-P (Nucleus)', color: 'amber', desc: 'Menor conjunto >= p' },
] as const

const COLOR_MAP: Record<string, { card: string; label: string; token: string; badge: string }> = {
  blue:   { card: 'border-blue-200',   label: 'text-blue-600',   token: 'bg-blue-50 border-blue-200 text-blue-700',     badge: 'bg-blue-100 text-blue-700' },
  green:  { card: 'border-green-200',  label: 'text-green-600',  token: 'bg-green-50 border-green-200 text-green-700',   badge: 'bg-green-100 text-green-700' },
  red:    { card: 'border-red-200',    label: 'text-red-600',    token: 'bg-red-50 border-red-200 text-red-700',         badge: 'bg-red-100 text-red-700' },
  purple: { card: 'border-purple-200', label: 'text-purple-600', token: 'bg-purple-50 border-purple-200 text-purple-700', badge: 'bg-purple-100 text-purple-700' },
  amber:  { card: 'border-amber-200',  label: 'text-amber-600',  token: 'bg-amber-50 border-amber-200 text-amber-700',   badge: 'bg-amber-100 text-amber-700' },
  cyan:   { card: 'border-cyan-200',   label: 'text-cyan-600',   token: 'bg-cyan-50 border-cyan-200 text-cyan-700',      badge: 'bg-cyan-100 text-cyan-700' },
}

// Detect repeated consecutive tokens
function findRepeats(tokens: string[]): Set<number> {
  const repeats = new Set<number>()
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i] === tokens[i - 1]) {
      repeats.add(i)
      repeats.add(i - 1)
    }
  }
  return repeats
}

const FALLBACK_STRATEGIES = ['Greedy', 'Temp. Baixa', 'Temp. Alta', 'Top-K', 'Top-P', 'Beam Search']

export default function SamplingComparison({ prompt = ['o', 'gato'] }: SamplingComparisonProps) {
  const samplingDemo = useSamplingDemo()

  useEffect(() => {
    samplingDemo.execute({ prompt, max_tokens: 8 })
  }, [prompt.join(' ')])

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Shuffle size={16} weight="duotone" className="text-violet-500" />
        <h4 className="text-sm font-semibold text-gray-800">Dashboard de Sampling</h4>
      </div>
      <p className="text-xs text-gray-500">
        O mesmo prompt "{prompt.join(' ')}" gera textos diferentes dependendo da estrategia de amostragem.
      </p>

      <ApiLoadingState
        loading={samplingDemo.loading}
        error={samplingDemo.error}
        compact
        fallback={
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {FALLBACK_STRATEGIES.map((nome) => (
              <div key={nome} className="glass-card p-4">
                <p className="text-xs font-semibold font-mono text-gray-500 mb-1.5">{nome}</p>
                <p className="text-xs text-gray-400">Dados indisponiveis offline</p>
              </div>
            ))}
          </div>
        }
      >
        {samplingDemo.data && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {STRATEGIES.map(({ key, label, color, desc }) => {
                const result = samplingDemo.data![key as keyof typeof samplingDemo.data] as Record<string, unknown> | undefined
                if (!result || Array.isArray(result)) return null
                const tokens = (result.tokens_gerados as string[] | undefined) ?? []
                const repeats = findRepeats(tokens)
                const colors = COLOR_MAP[color]

                return (
                  <div key={key} className={`glass-card p-4 border ${colors.card}`}>
                    <div className="flex items-center justify-between mb-2">
                      <p className={`text-xs font-semibold font-mono ${colors.label}`}>{label}</p>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-sm ${colors.badge}`}>
                        {tokens.length} tokens
                      </span>
                    </div>
                    <p className="text-[10px] text-gray-400 mb-2">{desc}</p>

                    {result.texto_gerado != null && (
                      <p className="text-xs font-mono text-gray-700 mb-2 leading-relaxed">
                        &quot;{String(result.texto_gerado)}&quot;
                      </p>
                    )}

                    <div className="flex flex-wrap gap-1">
                      {tokens.map((t, i) => (
                        <span
                          key={i}
                          className={`px-1 py-0.5 rounded-sm border text-[10px] font-mono ${
                            repeats.has(i)
                              ? 'bg-red-50 border-red-300 text-red-600 line-through'
                              : colors.token
                          }`}
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                )
              })}

              {/* Beam search */}
              {samplingDemo.data.beam_search && samplingDemo.data.beam_search.length > 0 && (
                <div className={`glass-card p-4 border ${COLOR_MAP.cyan.card}`}>
                  <div className="flex items-center justify-between mb-2">
                    <p className={`text-xs font-semibold font-mono ${COLOR_MAP.cyan.label}`}>Beam Search</p>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded-sm ${COLOR_MAP.cyan.badge}`}>
                      {samplingDemo.data.beam_search.length} beams
                    </span>
                  </div>
                  <p className="text-[10px] text-gray-400 mb-2">Busca em largura otimizada</p>
                  {samplingDemo.data.beam_search.map((beam, i) => {
                    const b = beam as Record<string, unknown>
                    return (
                      <div key={i} className="mb-1.5">
                        <p className="text-[10px] text-gray-400">Beam {i + 1}:</p>
                        <p className="text-xs font-mono text-gray-700">
                          &quot;{String(b.texto_gerado ?? '')}&quot;
                        </p>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {samplingDemo.data.explicacao && (
              <p className="text-xs text-gray-500 leading-relaxed">
                {samplingDemo.data.explicacao}
              </p>
            )}
          </div>
        )}
      </ApiLoadingState>
    </div>
  )
}
