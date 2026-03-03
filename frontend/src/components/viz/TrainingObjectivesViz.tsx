import { ArrowRight, MaskHappy, TextAlignLeft } from '@phosphor-icons/react'
import { useTrainingObjectives } from '../../api/hooks'
import ApiLoadingState from '../education/ApiLoadingState'

const FALLBACK = {
  frase_original: 'o gato dormiu no tapete',
  tokens: ['o', 'gato', 'dormiu', 'no', 'tapete'],
  mlm: {
    tokens_mascarados: ['o', '[MASK]', 'dormiu', '[MASK]', 'tapete'],
    indices_mascarados: [1, 3],
    predicoes: [
      { indice: 1, token_original: 'gato', top_predicoes: ['gato', 'cachorro', 'menino'] },
      { indice: 3, token_original: 'no', top_predicoes: ['no', 'na', 'ao'] },
    ],
  },
  clm: {
    predicoes_sequenciais: [
      { contexto: ['o'], alvo: 'gato', top_predicoes: ['gato', 'cachorro', 'menino'] },
      { contexto: ['o', 'gato'], alvo: 'dormiu', top_predicoes: ['dormiu', 'correu', 'sentou'] },
      { contexto: ['o', 'gato', 'dormiu'], alvo: 'no', top_predicoes: ['no', 'na', 'ao'] },
      { contexto: ['o', 'gato', 'dormiu', 'no'], alvo: 'tapete', top_predicoes: ['tapete', 'sofa', 'chao'] },
    ],
  },
  comparacao: 'MLM usa contexto bidirecional (ideal para compreensao), CLM usa contexto causal (ideal para geracao).',
}

type MLMData = typeof FALLBACK.mlm
type CLMData = typeof FALLBACK.clm

export default function TrainingObjectivesViz() {
  const objectives = useTrainingObjectives()

  const data = objectives.data ?? FALLBACK
  const mlm = data.mlm as MLMData
  const clm = data.clm as CLMData

  return (
    <div className="glass-card p-5 space-y-4">
      <h4 className="text-sm font-semibold text-gray-800">MLM vs CLM - Comparacao Visual</h4>
      <p className="text-xs text-gray-500">
        Dois objetivos de treinamento diferentes produzem modelos com capacidades distintas.
      </p>

      <ApiLoadingState
        loading={objectives.loading}
        error={objectives.error}
        compact
        fallback={
          <div className="text-xs text-gray-400 text-center py-4">
            Dados de exemplo - inicie o backend para resultados reais.
          </div>
        }
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* MLM Side */}
          <div className="space-y-3 p-4 rounded-sm bg-amber-50 border border-amber-200">
            <div className="flex items-center gap-2">
              <MaskHappy size={16} weight="duotone" className="text-amber-600" />
              <span className="text-xs font-semibold text-amber-700 font-mono">MLM - Masked Language Model</span>
            </div>
            <p className="text-[11px] text-amber-600">
              Mascara tokens aleatorios e preve usando contexto bidirecional (esquerda + direita).
            </p>

            {/* Masked sentence */}
            <div className="flex flex-wrap gap-1.5 p-3 rounded-sm bg-white border border-amber-100">
              {(mlm.tokens_mascarados ?? data.tokens).map((t, i) => {
                const isMasked = t === '[MASK]'
                return (
                  <span
                    key={i}
                    className={`px-2 py-1 rounded-sm border text-[11px] font-mono transition-all ${
                      isMasked
                        ? 'bg-amber-200 border-amber-300 text-amber-800 font-semibold'
                        : 'bg-white border-gray-200 text-gray-600'
                    }`}
                  >
                    {t}
                  </span>
                )
              })}
            </div>

            {/* Predictions */}
            {mlm.predicoes && (
              <div className="space-y-2">
                {mlm.predicoes.map((pred, i) => (
                  <div key={i} className="flex items-center gap-2 text-[11px]">
                    <span className="text-amber-500 font-mono">[MASK]</span>
                    <ArrowRight size={10} className="text-gray-400" />
                    <span className="font-mono text-amber-700 font-semibold">{pred.token_original}</span>
                    {pred.top_predicoes && (
                      <span className="text-gray-400 font-mono">
                        ({pred.top_predicoes.slice(0, 3).join(', ')})
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}

            <div className="text-[10px] text-amber-600 bg-amber-100 rounded-sm px-2 py-1">
              Modelos: BERT, RoBERTa, ALBERT. Uso: classificacao, NER, busca semantica.
            </div>
          </div>

          {/* CLM Side */}
          <div className="space-y-3 p-4 rounded-sm bg-blue-50 border border-blue-200">
            <div className="flex items-center gap-2">
              <TextAlignLeft size={16} weight="duotone" className="text-blue-600" />
              <span className="text-xs font-semibold text-blue-700 font-mono">CLM - Causal Language Model</span>
            </div>
            <p className="text-[11px] text-blue-600">
              Preve o proximo token usando apenas o contexto anterior (esquerda).
            </p>

            {/* Sequential predictions */}
            <div className="space-y-1.5 p-3 rounded-sm bg-white border border-blue-100">
              {clm.predicoes_sequenciais && clm.predicoes_sequenciais.map((pred, i) => (
                <div key={i} className="flex items-center gap-1 text-[11px] font-mono">
                  <div className="flex gap-0.5">
                    {pred.contexto.map((c, j) => (
                      <span key={j} className="px-1 py-0.5 bg-gray-100 border border-gray-200 rounded-sm text-gray-500">
                        {c}
                      </span>
                    ))}
                  </div>
                  <ArrowRight size={10} className="text-blue-400 flex-shrink-0" />
                  <span className="px-1.5 py-0.5 bg-blue-100 border border-blue-200 rounded-sm text-blue-700 font-semibold">
                    {pred.alvo}
                  </span>
                </div>
              ))}
            </div>

            <div className="text-[10px] text-blue-600 bg-blue-100 rounded-sm px-2 py-1">
              Modelos: GPT, Llama, Claude. Uso: geracao de texto, chatbots, codigo.
            </div>
          </div>
        </div>

        {/* Comparison note */}
        <div className="p-3 rounded-sm bg-gray-50 border border-gray-200">
          <p className="text-xs text-gray-600 leading-relaxed">
            {data.comparacao}
          </p>
        </div>
      </ApiLoadingState>
    </div>
  )
}
