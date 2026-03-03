import { useState } from 'react'
import { Lightning, ArrowRight } from '@phosphor-icons/react'
import { useGenerate } from '../../api/hooks'
import ApiLoadingState from '../education/ApiLoadingState'

const FRASES_EXEMPLO = [
  'O gato sentou no',
  'A professora explicou a',
  'O sol brilha no',
  'Eu gosto de comer',
  'O cachorro correu para',
]

export default function PredictNextWord() {
  const [frase, setFrase] = useState(FRASES_EXEMPLO[0])
  const [palpite, setPalpite] = useState('')
  const [mostrarResultado, setMostrarResultado] = useState(false)
  const generate = useGenerate()

  const handlePrever = () => {
    if (!frase.trim()) return
    const tokens = frase.trim().split(/\s+/)
    generate.execute({
      prompt: tokens,
      estrategia: 'greedy',
      max_tokens: 1,
    })
    setMostrarResultado(true)
  }

  const handleReset = () => {
    setMostrarResultado(false)
    setPalpite('')
    generate.reset()
  }

  const top5 = generate.data?.historico_probabilidades?.[0]?.top_5_tokens ?? []
  const tokenEscolhido = generate.data?.tokens_gerados?.[0] ?? ''
  const palpiteCorreto = palpite.trim().toLowerCase() === tokenEscolhido.trim().toLowerCase()

  return (
    <div className="glass-card p-6 space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <Lightning size={16} weight="duotone" className="text-amber-500" />
        <h3 className="text-sm font-semibold text-gray-800">Jogo: Preveja a Proxima Palavra</h3>
      </div>

      <p className="text-xs text-gray-500 leading-relaxed">
        Tente adivinhar qual palavra o modelo escolheria para completar a frase. Depois veja o que ele realmente diria!
      </p>

      {/* Frases sugeridas */}
      <div className="flex flex-wrap gap-1.5">
        {FRASES_EXEMPLO.map((f) => (
          <button
            key={f}
            onClick={() => { setFrase(f); handleReset() }}
            className={`px-2 py-1 rounded-sm border text-[11px] transition-all ${
              frase === f
                ? 'bg-amber-50 border-amber-200 text-amber-700'
                : 'bg-gray-100 border-gray-200 text-gray-500 hover:border-gray-300'
            }`}
          >
            {f}...
          </button>
        ))}
      </div>

      {/* Input da frase */}
      <div>
        <label className="text-xs font-medium text-gray-600 mb-1 block">Frase incompleta</label>
        <input
          type="text"
          value={frase}
          onChange={(e) => { setFrase(e.target.value); handleReset() }}
          className="w-full bg-white border border-gray-300 rounded-sm px-3 py-2 text-sm text-gray-900 font-mono focus:outline-none focus:border-amber-400 transition-colors"
          placeholder="Digite uma frase incompleta..."
        />
      </div>

      {/* Input do palpite */}
      <div>
        <label className="text-xs font-medium text-gray-600 mb-1 block">Seu palpite para a proxima palavra</label>
        <div className="flex gap-2">
          <input
            type="text"
            value={palpite}
            onChange={(e) => setPalpite(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePrever()}
            className="flex-1 bg-white border border-gray-300 rounded-sm px-3 py-2 text-sm text-gray-900 font-mono focus:outline-none focus:border-amber-400 transition-colors"
            placeholder="Qual palavra vem a seguir?"
            disabled={mostrarResultado}
          />
          <button
            onClick={mostrarResultado ? handleReset : handlePrever}
            disabled={generate.loading}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-sm bg-amber-500 hover:bg-amber-400 disabled:bg-gray-300 text-white text-xs font-medium transition-colors"
          >
            {mostrarResultado ? 'Tentar de novo' : (
              <>Ver resultado <ArrowRight size={12} weight="bold" /></>
            )}
          </button>
        </div>
      </div>

      {/* Resultado */}
      {mostrarResultado && (
        <ApiLoadingState
          loading={generate.loading}
          error={generate.error}
          compact
          fallback={
            <div className="text-center py-4 text-xs text-gray-500">
              Inicie o backend para jogar com o modelo real.
            </div>
          }
        >
          {generate.data && (
            <div className="space-y-3 animate-slide-up">
              {/* Comparacao */}
              {palpite.trim() && (
                <div className={`p-3 rounded-sm border text-sm ${
                  palpiteCorreto
                    ? 'bg-green-50 border-green-200 text-green-700'
                    : 'bg-orange-50 border-orange-200 text-orange-700'
                }`}>
                  {palpiteCorreto
                    ? 'Acertou! Voce pensou como o modelo!'
                    : `Voce disse "${palpite}", o modelo disse "${tokenEscolhido}". Ambos sao validos!`}
                </div>
              )}

              {/* Top 5 candidatos */}
              <div>
                <p className="text-xs font-medium text-gray-600 mb-2">Top 5 candidatos do modelo:</p>
                <div className="space-y-1.5">
                  {top5.map((t, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs font-mono text-gray-500 w-4 text-right">{i + 1}.</span>
                      <span className={`text-xs font-mono font-medium w-20 ${
                        i === 0 ? 'text-amber-600' : 'text-gray-600'
                      }`}>
                        {t.token}
                      </span>
                      <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            i === 0 ? 'bg-amber-400' : 'bg-gray-300'
                          }`}
                          style={{ width: `${Math.max(2, t.probabilidade * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-gray-400 w-12 text-right">
                        {(t.probabilidade * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Frase completa */}
              <div className="p-3 rounded-sm bg-gray-50 border border-gray-200">
                <p className="text-xs text-gray-500 mb-1">Frase completa:</p>
                <p className="text-sm font-mono text-gray-800">
                  {frase} <span className="text-amber-600 font-semibold">{tokenEscolhido}</span>
                </p>
              </div>
            </div>
          )}
        </ApiLoadingState>
      )}
    </div>
  )
}
