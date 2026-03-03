import { useState, useEffect } from 'react'
import { Play, Pause, CaretLeft, CaretRight, ArrowCounterClockwise } from '@phosphor-icons/react'
import { useBPESteps } from '../../api/hooks'
import ApiLoadingState from '../education/ApiLoadingState'

interface BPEAnimationProps {
  texto?: string
}

const FALLBACK_PASSOS = [
  { passo: 1, par: ['t', 'a'] as [string, string], frequencia: 3, tokens_atuais: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u', ' ', 'n', 'o', ' ', 'ta', 'p', 'e', 't', 'e'], vocabulario_tamanho: 21 },
  { passo: 2, par: ['n', 'o'] as [string, string], frequencia: 2, tokens_atuais: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u', ' ', 'no', ' ', 'ta', 'p', 'e', 't', 'e'], vocabulario_tamanho: 20 },
]

export default function BPEAnimation({ texto = 'O gato dormiu no tapete' }: BPEAnimationProps) {
  const bpeSteps = useBPESteps()
  const [passoAtual, setPassoAtual] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    if (texto.trim()) {
      bpeSteps.execute({ texto, num_merges: 10 })
      setPassoAtual(0)
      setPlaying(false)
    }
  }, [texto])

  const passos = bpeSteps.data?.passos ?? FALLBACK_PASSOS
  const totalPassos = passos.length

  // Autoplay
  useEffect(() => {
    if (!playing || totalPassos === 0) return
    const timer = setInterval(() => {
      setPassoAtual((prev) => {
        if (prev >= totalPassos - 1) {
          setPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, 1500)
    return () => clearInterval(timer)
  }, [playing, totalPassos])

  const passoData = passos[passoAtual]
  if (!passoData) return null

  // Build vocabulary list up to current step
  const vocabAcumulado: string[] = []
  for (let i = 0; i <= passoAtual; i++) {
    const p = passos[i]
    if (p) vocabAcumulado.push(p.par.join(''))
  }

  return (
    <div className="glass-card p-5 space-y-4">
      <h4 className="text-sm font-semibold text-gray-800">Animacao BPE Passo a Passo</h4>
      <p className="text-xs text-gray-500">
        Veja como o BPE funde pares de tokens iterativamente, construindo um vocabulario de subpalavras.
      </p>

      <ApiLoadingState
        loading={bpeSteps.loading}
        error={bpeSteps.error}
        compact
        fallback={
          <div className="text-xs text-gray-400 text-center py-4">
            Dados de exemplo - inicie o backend para dados reais.
          </div>
        }
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Main area: tokens */}
          <div className="md:col-span-2 space-y-3">
            {/* Step info */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-gray-500">
                  Passo {passoData.passo} / {totalPassos}
                </span>
                <span className="text-xs text-emerald-600 font-medium">
                  Fundir "{passoData.par[0]}" + "{passoData.par[1]}"
                </span>
              </div>
              <span className="text-[10px] text-gray-400 font-mono">
                freq: {passoData.frequencia}x
              </span>
            </div>

            {/* Current tokens */}
            <div className="flex flex-wrap gap-1 min-h-[48px] p-3 rounded-sm bg-gray-50 border border-gray-200">
              {passoData.tokens_atuais.map((t, i) => {
                const merged = t === passoData.par.join('')
                return (
                  <span
                    key={i}
                    className={`px-1.5 py-0.5 rounded-sm border text-[11px] font-mono transition-all duration-300 ${
                      merged
                        ? 'bg-emerald-50 border-emerald-300 text-emerald-700 font-semibold ring-1 ring-emerald-200'
                        : 'bg-white border-gray-200 text-gray-600'
                    }`}
                  >
                    {t === ' ' ? '\u00B7' : t}
                  </span>
                )
              })}
            </div>

            {/* Stats */}
            <div className="flex items-center gap-4 text-[11px] text-gray-500">
              <span>Tokens: <span className="font-mono text-gray-700">{passoData.tokens_atuais.length}</span></span>
              <span>Vocabulario: <span className="font-mono text-gray-700">{passoData.vocabulario_tamanho}</span></span>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => { setPassoAtual(0); setPlaying(false) }}
                className="p-1.5 rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors"
                title="Resetar"
              >
                <ArrowCounterClockwise size={14} weight="regular" />
              </button>
              <button
                onClick={() => setPassoAtual(Math.max(0, passoAtual - 1))}
                disabled={passoAtual === 0}
                className="p-1.5 rounded-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-40 text-gray-600 transition-colors"
              >
                <CaretLeft size={14} weight="bold" />
              </button>
              <button
                onClick={() => setPlaying(!playing)}
                className={`p-1.5 rounded-sm transition-colors ${
                  playing
                    ? 'bg-amber-100 text-amber-600 hover:bg-amber-200'
                    : 'bg-emerald-100 text-emerald-600 hover:bg-emerald-200'
                }`}
              >
                {playing ? <Pause size={14} weight="fill" /> : <Play size={14} weight="fill" />}
              </button>
              <button
                onClick={() => setPassoAtual(Math.min(totalPassos - 1, passoAtual + 1))}
                disabled={passoAtual >= totalPassos - 1}
                className="p-1.5 rounded-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-40 text-gray-600 transition-colors"
              >
                <CaretRight size={14} weight="bold" />
              </button>

              {/* Progress bar */}
              <div className="flex-1 bg-gray-200 rounded-full h-1.5 ml-2">
                <div
                  className="bg-emerald-500 h-full rounded-full transition-all duration-300"
                  style={{ width: `${((passoAtual + 1) / totalPassos) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Sidebar: vocabulary */}
          <div className="space-y-2">
            <p className="text-xs font-medium text-gray-600">Vocabulario Aprendido</p>
            <div className="max-h-48 overflow-y-auto space-y-1 p-2 rounded-sm bg-gray-50 border border-gray-200">
              {vocabAcumulado.length === 0 ? (
                <p className="text-[10px] text-gray-400 text-center py-2">Nenhum merge ainda</p>
              ) : (
                vocabAcumulado.map((v, i) => (
                  <div
                    key={i}
                    className={`flex items-center justify-between px-2 py-1 rounded-sm text-[11px] font-mono transition-all ${
                      i === passoAtual
                        ? 'bg-emerald-50 border border-emerald-200 text-emerald-700'
                        : 'text-gray-600'
                    }`}
                  >
                    <span>{v}</span>
                    <span className="text-gray-400">#{i + 1}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </ApiLoadingState>
    </div>
  )
}
