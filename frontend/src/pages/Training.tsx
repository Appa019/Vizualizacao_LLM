import { useState, useEffect, useMemo } from 'react'
import { Student, Play, ArrowCounterClockwise } from '@phosphor-icons/react'
import { useTrainStep, useLossSurface, useGradientDescent, useTrainingObjectives } from '../api/hooks'
import LossSurface from '../components/viz/LossSurface'
import PlotlyChart from '../components/viz/PlotlyChart'
import StepByStep from '../components/education/StepByStep'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import Slider from '../components/ui/Slider'
import ApiLoadingState from '../components/education/ApiLoadingState'

// ─── Fallback: Loss Surface (30x30 quadratic bowl L = w1^2 + w2^2) ─────────
function buildLossSurfaceFallback() {
  const size = 30
  const range: number[] = []
  for (let i = 0; i < size; i++) {
    range.push(-3 + (6 / (size - 1)) * i)
  }
  const grid: number[][] = []
  for (let i = 0; i < size; i++) {
    const row: number[] = []
    for (let j = 0; j < size; j++) {
      row.push(range[i] * range[i] + range[j] * range[j])
    }
    grid.push(row)
  }
  return { w1_valores: range, w2_valores: range, loss_grid: grid, ponto_otimo: { w1: 0, w2: 0, loss: 0 }, explicacao: 'Superficie quadratica L = w1^2 + w2^2. O minimo global esta em (0, 0).' }
}

// ─── Fallback: Gradient Descent trajectory ──────────────────────────────────
const FALLBACK_GD_PASSOS = [
  { iteracao: 0, w1: 2.5, w2: 2.5, loss: 12.5, gradiente_w1: 5.0, gradiente_w2: 5.0 },
  { iteracao: 1, w1: 2.0, w2: 2.0, loss: 8.0, gradiente_w1: 4.0, gradiente_w2: 4.0 },
  { iteracao: 2, w1: 1.6, w2: 1.6, loss: 5.12, gradiente_w1: 3.2, gradiente_w2: 3.2 },
  { iteracao: 3, w1: 1.28, w2: 1.28, loss: 3.28, gradiente_w1: 2.56, gradiente_w2: 2.56 },
  { iteracao: 4, w1: 1.0, w2: 1.0, loss: 2.0, gradiente_w1: 2.0, gradiente_w2: 2.0 },
  { iteracao: 5, w1: 0.8, w2: 0.8, loss: 1.28, gradiente_w1: 1.6, gradiente_w2: 1.6 },
  { iteracao: 6, w1: 0.6, w2: 0.6, loss: 0.72, gradiente_w1: 1.2, gradiente_w2: 1.2 },
  { iteracao: 7, w1: 0.4, w2: 0.4, loss: 0.32, gradiente_w1: 0.8, gradiente_w2: 0.8 },
  { iteracao: 8, w1: 0.2, w2: 0.2, loss: 0.08, gradiente_w1: 0.4, gradiente_w2: 0.4 },
  { iteracao: 9, w1: 0.1, w2: 0.1, loss: 0.02, gradiente_w1: 0.2, gradiente_w2: 0.2 },
]

const FALLBACK_GRADIENT_DESCENT = {
  passos: FALLBACK_GD_PASSOS,
  taxa_aprendizado: 0.1,
  loss_inicial: 12.5,
  loss_final: 0.02,
  convergiu: true,
  reducao_percentual: 99.84,
  explicacao: 'Gradient descent convergiu de loss 12.5 para 0.02, uma reducao de 99.84%.',
}

// ─── Fallback: Training Objectives (MLM vs CLM) ────────────────────────────
const FALLBACK_OBJECTIVES = {
  frase_original: 'O gato dormiu no sofa',
  tokens: ['O', 'gato', 'dormiu', 'no', 'sofa'],
  mlm: {
    tokens_mascarados: ['O', '[MASK]', 'dormiu', 'no', '[MASK]'],
    predicoes: [
      { posicao: 1, token_original: 'gato', top_predicoes: [
        { token: 'gato', probabilidade: 0.42 },
        { token: 'cachorro', probabilidade: 0.18 },
        { token: 'menino', probabilidade: 0.09 },
      ]},
      { posicao: 4, token_original: 'sofa', top_predicoes: [
        { token: 'sofa', probabilidade: 0.35 },
        { token: 'chao', probabilidade: 0.22 },
        { token: 'tapete', probabilidade: 0.11 },
      ]},
    ],
  },
  clm: {
    contextos: [
      { contexto: ['O'], proximo_token: 'gato', top_predicoes: [
        { token: 'gato', probabilidade: 0.25 },
        { token: 'menino', probabilidade: 0.15 },
      ]},
      { contexto: ['O', 'gato'], proximo_token: 'dormiu', top_predicoes: [
        { token: 'dormiu', probabilidade: 0.30 },
        { token: 'comeu', probabilidade: 0.12 },
      ]},
    ],
  },
  comparacao: 'MLM (como BERT) mascara tokens e usa contexto bidirecional para preve-los, sendo ideal para tarefas de compreensao. CLM (como GPT) preve o proximo token usando apenas o contexto anterior, sendo naturalmente gerativo. Ambos usam cross-entropy loss, mas a direcionalidade do contexto define as capacidades do modelo.',
}

export default function Training() {
  const [learningRate, setLearningRate] = useState(0.01)
  const [gdLr, setGdLr] = useState(0.1)

  const trainStep = useTrainStep()
  const lossSurface = useLossSurface()
  const gradientDescent = useGradientDescent()
  const objectives = useTrainingObjectives()

  useEffect(() => {
    lossSurface.execute({ resolucao: 30 })
    gradientDescent.execute({ taxa_aprendizado: gdLr, num_iteracoes: 50 })
  }, [])

  useEffect(() => {
    gradientDescent.execute({ taxa_aprendizado: gdLr, num_iteracoes: 50 })
  }, [gdLr])

  const handleTrainStep = () => {
    trainStep.execute({ taxa_aprendizado: learningRate })
  }

  const handleReset = () => {
    trainStep.execute({ taxa_aprendizado: learningRate, resetar: true })
  }

  const fallbackLossSurface = useMemo(() => buildLossSurfaceFallback(), [])

  const surfaceData = lossSurface.data?.loss_grid
  const trajectoryPoints: [number, number, number][] | undefined = gradientDescent.data?.passos.map(
    (p) => [p.w1 - 3, p.loss, p.w2 - 3] as [number, number, number]
  )

  const fallbackTrajectoryPoints: [number, number, number][] = FALLBACK_GD_PASSOS.map(
    (p) => [p.w1 - 3, p.loss, p.w2 - 3] as [number, number, number]
  )

  const lossHistory = trainStep.data?.historico_loss ?? []

  const buildObjSteps = (data: { frase_original: string; tokens: string[]; mlm: Record<string, unknown> }) => [
    {
      title: 'Frase Original',
      description: `"${data.frase_original}" - tokenizada em [${data.tokens.join(', ')}]`,
      content: (
        <div className="flex flex-wrap gap-1.5">
          {data.tokens.map((t, i) => (
            <span key={i} className="px-2 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-mono text-gray-700">{t}</span>
          ))}
        </div>
      ),
    },
    {
      title: 'MLM - Masked Language Modeling (BERT)',
      description: 'Mascara tokens aleatorios e os preve usando contexto bidirecional. Ideal para compreensao.',
      content: (
        <div className="text-xs font-mono text-gray-600 bg-white rounded-sm p-3 space-y-1">
          <p>Entrada: {JSON.stringify((data.mlm as Record<string, unknown>).tokens_mascarados ?? [])}</p>
          <p>Mascara: tokens [MASK] devem ser previstos usando todo o contexto</p>
        </div>
      ),
      whyItMatters: 'MLM entende a frase inteira - "O [MASK] dormiu" pode prever "gato" usando "dormiu" a direita.',
    },
    {
      title: 'CLM - Causal Language Modeling (GPT)',
      description: 'Preve o proximo token dado apenas o contexto anterior. Naturalmente gerativo.',
      content: (
        <div className="text-xs font-mono text-gray-600 bg-white rounded-sm p-3 space-y-1">
          <p>Cada posicao so ve tokens anteriores (mascara causal)</p>
        </div>
      ),
      whyItMatters: 'CLM e naturalmente gerativo - e assim que ChatGPT, Gemini e outros chatbots geram texto.',
    },
  ]

  const objSteps = objectives.data ? buildObjSteps(objectives.data) : []

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Student size={12} weight="duotone" />
          Modulo 07
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">Treinamento</h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          O treinamento e o processo de ajustar bilhoes de parametros para minimizar uma funcao de loss.
          Entenda como gradient descent navega a paisagem de otimizacao e como diferentes
          objetivos (MLM vs CLM) ensinam o modelo.
        </p>
      </section>

      <FormulaBlock
        formula="\\mathcal{L} = -\\sum_{i=1}^{V} y_i \\log(\\hat{y}_i)"
        variables={[
          { symbol: '\\mathcal{L}', color: '#ef4444', label: 'Loss', description: 'Funcao de custo - queremos minimizar' },
          { symbol: 'y_i', color: '#22c55e', label: 'Alvo', description: 'Distribuicao alvo (one-hot do token correto)' },
          { symbol: '\\hat{y}_i', color: '#3b82f6', label: 'Predicao', description: 'Probabilidade prevista pelo modelo' },
          { symbol: 'V', color: '#f59e0b', label: 'Vocabulario', description: 'Tamanho do vocabulario' },
        ]}
        size="lg"
      />

      <EducationalViz
        title="Superficie de Loss 3D"
        icon={<Student size={18} weight="duotone" />}
        caption="A superficie mostra como o loss varia para diferentes pesos. A bolinha vermelha mostra o gradient descent descendo ate o minimo."
        formula={{
          formula: "w_{t+1} = w_t - \\eta \\nabla \\mathcal{L}(w_t)",
          variables: [
            { symbol: 'w_t', color: '#3b82f6', label: 'Pesos atuais', description: 'Valor dos pesos no passo t' },
            { symbol: '\\eta', color: '#f59e0b', label: 'Learning rate', description: 'Taxa de aprendizado' },
            { symbol: '\\nabla', color: '#ef4444', label: 'Gradiente', description: 'Direcao de maior subida' },
          ],
        }}
      >
        <div className="space-y-4">
          <Slider value={gdLr} onChange={setGdLr} min={0.01} max={1.0} step={0.01} label="Learning Rate do Gradient Descent" />
          <ApiLoadingState
            loading={lossSurface.loading}
            error={lossSurface.error}
            fallback={
              <LossSurface
                surfaceData={fallbackLossSurface.loss_grid}
                trajectoryPoints={fallbackTrajectoryPoints}
                height={450}
                gridSize={30}
              />
            }
          >
            <LossSurface
              surfaceData={surfaceData}
              trajectoryPoints={trajectoryPoints}
              height={450}
              gridSize={lossSurface.data ? lossSurface.data.w1_valores.length : 30}
            />
          </ApiLoadingState>
          <ApiLoadingState
            loading={gradientDescent.loading}
            error={gradientDescent.error}
            fallback={
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Loss Inicial</p>
                  <p className="text-sm font-mono text-red-400">{FALLBACK_GRADIENT_DESCENT.loss_inicial}</p>
                </div>
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Loss Final</p>
                  <p className="text-sm font-mono text-green-400">{FALLBACK_GRADIENT_DESCENT.loss_final}</p>
                </div>
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Reducao</p>
                  <p className="text-sm font-mono text-blue-400">{FALLBACK_GRADIENT_DESCENT.reducao_percentual}%</p>
                </div>
              </div>
            }
          >
            {gradientDescent.data && (
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Loss Inicial</p>
                  <p className="text-sm font-mono text-red-400">{gradientDescent.data.loss_inicial}</p>
                </div>
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Loss Final</p>
                  <p className="text-sm font-mono text-green-400">{gradientDescent.data.loss_final}</p>
                </div>
                <div className="text-center p-2 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Reducao</p>
                  <p className="text-sm font-mono text-blue-400">{gradientDescent.data.reducao_percentual}%</p>
                </div>
              </div>
            )}
          </ApiLoadingState>
        </div>
      </EducationalViz>

      <EducationalViz
        title="Treino ao Vivo"
        icon={<Student size={18} weight="duotone" />}
        caption="Cada clique executa forward pass, calcula o loss e atualiza os pesos via backpropagation."
      >
        <div className="space-y-4">
          <Slider value={learningRate} onChange={setLearningRate} min={0.001} max={0.1} step={0.001} label="Learning Rate" />
          <div className="flex gap-2">
            <button onClick={handleTrainStep} disabled={trainStep.loading} className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-blue-600 hover:bg-blue-500 disabled:bg-gray-300 text-white text-sm font-medium transition-colors">
              <Play size={14} weight="fill" /> Treinar Passo
            </button>
            <button onClick={handleReset} className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm transition-colors">
              <ArrowCounterClockwise size={14} weight="regular" /> Resetar
            </button>
          </div>
          {trainStep.data && (
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center p-3 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Passo</p>
                  <p className="text-lg font-mono text-blue-400">{trainStep.data.passo}</p>
                </div>
                <div className="text-center p-3 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Loss</p>
                  <p className="text-lg font-mono text-red-400">{trainStep.data.loss}</p>
                </div>
                <div className="text-center p-3 rounded-sm bg-gray-100">
                  <p className="text-xs text-gray-500">Acuracia</p>
                  <p className="text-lg font-mono text-green-400">{(trainStep.data.acuracia * 100).toFixed(0)}%</p>
                </div>
              </div>
              {lossHistory.length > 1 && (
                <PlotlyChart
                  data={[{ type: 'scatter', y: lossHistory, mode: 'lines', line: { color: '#ef4444', width: 2 }, name: 'Loss' } as unknown as Plotly.Data]}
                  layout={{ title: { text: 'Historico de Loss', font: { color: '#374151', size: 13 } }, xaxis: { title: { text: 'Passo' } }, yaxis: { title: { text: 'Loss' } }, margin: { t: 40, r: 20, b: 50, l: 60 } }}
                  height={250}
                />
              )}
              <div className="glass-card p-4">
                <p className="text-xs font-medium text-gray-600 mb-2">Norma dos Gradientes</p>
                <div className="flex flex-wrap gap-3">
                  {Object.entries(trainStep.data.gradientes_norma).map(([k, v]) => (
                    <div key={k} className="text-center">
                      <p className="text-[10px] text-gray-400 font-mono">{k}</p>
                      <p className="text-xs font-mono text-amber-400">{v}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </EducationalViz>

      <ApiLoadingState
        loading={objectives.loading}
        error={objectives.error}
        fallback={
          <>
            <StepByStep steps={buildObjSteps(FALLBACK_OBJECTIVES)} title="MLM vs CLM - Objetivos de Treinamento" />
            <div className="glass-card p-5">
              <p className="text-sm text-gray-700 leading-relaxed">{FALLBACK_OBJECTIVES.comparacao}</p>
            </div>
          </>
        }
      >
        {objSteps.length > 0 && <StepByStep steps={objSteps} title="MLM vs CLM - Objetivos de Treinamento" />}
        {objectives.data && (
          <div className="glass-card p-5">
            <p className="text-sm text-gray-700 leading-relaxed">{objectives.data.comparacao}</p>
          </div>
        )}
      </ApiLoadingState>

      <WhyItMatters>
        <p>
          O treinamento e onde a "inteligencia" do modelo e criada. Bilhoes de parametros
          sao ajustados iterativamente para capturar padroes estatisticos da linguagem.
        </p>
        <p className="mt-2">
          A escolha do objetivo (MLM vs CLM) determina as capacidades: BERT (MLM) e melhor em
          compreensao, GPT (CLM) e melhor em geracao.
        </p>
      </WhyItMatters>
    </div>
  )
}
