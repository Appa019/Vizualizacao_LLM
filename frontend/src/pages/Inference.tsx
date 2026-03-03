import { useState, useEffect } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Lightning, Play } from '@phosphor-icons/react'
import { useGenerate, useTemperatureDemo, useSamplingDemo } from '../api/hooks'
import type { LayoutContext } from '../components/layout/Layout'
import ConteudoAdaptavel from '../components/education/ConteudoAdaptavel'
import TokenStream from '../components/viz/TokenStream'
import PlotlyChart from '../components/viz/PlotlyChart'
import StepByStep from '../components/education/StepByStep'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import Slider from '../components/ui/Slider'
import ApiLoadingState from '../components/education/ApiLoadingState'
import InferencePipeline3D from '../components/viz/InferencePipeline3D'

// ─── Dados fallback quando o backend esta offline ────────────────────────────

const FALLBACK_TOKENS = ['o', 'gato', 'sentou', 'no', 'tapete']
const FALLBACK_PROBS = [
  [{ token: 'gato', probabilidade: 0.32 }, { token: 'cachorro', probabilidade: 0.18 }],
  [{ token: 'sentou', probabilidade: 0.28 }, { token: 'dormiu', probabilidade: 0.22 }],
  [{ token: 'no', probabilidade: 0.41 }, { token: 'na', probabilidade: 0.19 }],
  [{ token: 'tapete', probabilidade: 0.35 }, { token: 'sofa', probabilidade: 0.20 }],
  [{ token: '.', probabilidade: 0.45 }, { token: 'vermelho', probabilidade: 0.15 }],
]

const STRATEGY_OPTIONS = [
  { value: 'greedy', label: 'Greedy', desc: 'Sempre escolhe o token mais provavel' },
  { value: 'temperatura', label: 'Temperatura', desc: 'Controla a aleatoriedade via temperatura' },
  { value: 'top_k', label: 'Top-K', desc: 'Amostra dos K tokens mais provaveis' },
  { value: 'top_p', label: 'Top-P (Nucleus)', desc: 'Amostra do menor conjunto com prob >= p' },
]

// ─── Componente principal ────────────────────────────────────────────────────

export default function Inference() {
  const { nivelConhecimento } = useOutletContext<LayoutContext>()
  const [prompt, setPrompt] = useState('o gato')
  const [estrategia, setEstrategia] = useState('greedy')
  const [temperatura, setTemperatura] = useState(1.0)
  const [maxTokens, setMaxTokens] = useState(10)

  const generate = useGenerate()
  const tempDemo = useTemperatureDemo()
  const samplingDemo = useSamplingDemo()

  // Buscar dados de demonstracao ao montar
  useEffect(() => {
    tempDemo.execute({ temperaturas: [0.1, 0.5, 1.0, 1.5, 2.0] })
    samplingDemo.execute({ prompt: prompt.split(/\s+/), max_tokens: 5 })
  }, [])

  const handleGenerate = () => {
    const promptTokens = prompt.trim().split(/\s+/)
    generate.execute({
      prompt: promptTokens,
      estrategia,
      max_tokens: maxTokens,
      temperatura,
    })
  }

  // Dados para o TokenStream
  const generatedTokens = generate.data?.tokens_gerados ?? FALLBACK_TOKENS
  const tokenProbs = generate.data?.historico_probabilidades?.map(
    (h) => h.top_5_tokens.map((t) => t.probabilidade)
  )

  // Dados para o grafico de temperatura
  const tempChartData: Plotly.Data[] = tempDemo.data
    ? tempDemo.data.distribuicoes.map((dist) => ({
        type: 'bar' as const,
        name: `T=${dist.temperatura}`,
        x: tempDemo.data!.tokens,
        y: dist.probabilidades,
        hovertemplate: '%{x}: %{y:.3f}<extra>T=' + dist.temperatura + '</extra>',
      }))
    : []

  // StepByStep passos
  const inferenceSteps = [
    {
      title: '1. Contexto',
      description: 'O modelo recebe a sequencia de tokens de entrada como contexto inicial.',
      content: (
        <div className="flex flex-wrap gap-1.5">
          {prompt.split(/\s+/).map((t, i) => (
            <span key={i} className="px-2.5 py-1 bg-gray-100 border border-gray-300 rounded text-xs font-mono text-gray-700">
              {t}
            </span>
          ))}
        </div>
      ),
    },
    {
      title: '2. Forward Pass',
      description: 'Os tokens passam por todas as camadas do Transformer: embeddings, atencao multi-head e feed-forward networks.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-white rounded-sm p-3">
          Embedding → [N camadas x (Self-Attention → FFN)] → Projecao final
        </div>
      ),
      whyItMatters: 'O forward pass e altamente paralelizavel - todos os tokens sao processados simultaneamente, diferente de RNNs.',
    },
    {
      title: '3. Distribuicao de Probabilidade',
      description: 'A ultima camada projeta o estado oculto para o tamanho do vocabulario e aplica softmax para obter probabilidades.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-white rounded-sm p-3">
          logits = W_vocab @ hidden_state → probabilidades = softmax(logits / T)
        </div>
      ),
    },
    {
      title: '4. Amostragem',
      description: 'Uma estrategia de decodificacao seleciona o proximo token: greedy (mais provavel), temperatura, top-k ou top-p.',
      content: (
        <div className="grid grid-cols-2 gap-2">
          {STRATEGY_OPTIONS.map((s) => (
            <div key={s.value} className="text-xs bg-white rounded-sm p-2">
              <span className="text-cyan-400 font-mono font-medium">{s.label}</span>
              <p className="text-gray-500 mt-0.5">{s.desc}</p>
            </div>
          ))}
        </div>
      ),
      whyItMatters: 'A escolha da estrategia afeta drasticamente a qualidade: greedy e deterministico mas repetitivo, top-p gera textos mais naturais.',
    },
    {
      title: '5. Append e Repetir',
      description: 'O token selecionado e adicionado a sequencia e o processo se repete ate atingir o limite de tokens ou um token de parada.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-white rounded-sm p-3 space-y-1">
          <p>Iteracao 1: [o, gato] → "sentou"</p>
          <p>Iteracao 2: [o, gato, sentou] → "no"</p>
          <p>Iteracao 3: [o, gato, sentou, no] → "tapete"</p>
        </div>
      ),
      whyItMatters: 'Esse loop autoregressivo e como ChatGPT, Gemini e outros LLMs geram texto - um token por vez.',
    },
  ]

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Header */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Lightning size={12} weight="duotone" />
          Modulo 08
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Inferencia
        </h2>
        <ConteudoAdaptavel
          avancado={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              Durante a inferencia, o modelo gera texto um token por vez. A cada passo, ele
              produz uma distribuicao de probabilidade sobre o vocabulario inteiro e seleciona
              o proximo token com base em uma estrategia de decodificacao.
            </p>
          }
          iniciante={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              Inferencia e o autocompletar do celular turbinado. Voce digita 'Eu quero' e o modelo calcula a probabilidade de cada palavra ser a proxima: 'comer' 30%, 'ir' 25%, 'dormir' 15%... Depois escolhe uma e repete o processo ate formar uma frase completa.
            </p>
          }
        />
      </section>

      {/* Explicacao iniciante antes da formula */}
      <ConteudoAdaptavel
        avancado={null}
        iniciante={
          <section className="glass-card p-5 bg-green-50 border-green-200">
            <h4 className="text-sm font-semibold text-green-700 mb-2">Greedy vs Sampling - qual a diferenca?</h4>
            <p className="text-xs text-green-600 leading-relaxed">
              Greedy sempre escolhe a palavra mais provavel - e seguro mas chato, como sempre pedir o mesmo prato no restaurante. Sampling sorteia baseado nas probabilidades - e mais criativo, como experimentar pratos novos. A temperatura controla o quanto voce e aventureiro: baixa = seguro, alta = ousado.
            </p>
          </section>
        }
      />

      {/* Formula da temperatura */}
      <ConteudoAdaptavel
        avancado={
          <FormulaBlock
            formula="p_i = \\frac{e^{z_i/T}}{\\sum_j e^{z_j/T}}"
            variables={[
              { symbol: 'p_i', color: '#22d3ee', label: 'Probabilidade', description: 'Probabilidade ajustada do token i' },
              { symbol: 'z_i', color: '#3b82f6', label: 'Logit', description: 'Logit bruto do token i antes do softmax' },
              { symbol: 'T', color: '#ef4444', label: 'Temperatura', description: 'Controla a aleatoriedade: T<1 concentra, T>1 espalha' },
            ]}
            size="lg"
          />
        }
        iniciante={null}
      />

      {/* Controles de geracao */}
      <EducationalViz
        title="Geracao Interativa"
        icon={<Lightning size={18} weight="duotone" />}
        caption="Digite um prompt, escolha a estrategia e veja o modelo gerar tokens um a um com suas probabilidades."
      >
        <div className="space-y-4">
          {/* Input do prompt */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">
              Prompt de entrada
            </label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-white border border-gray-300 rounded-sm px-4 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-cyan-500/50 transition-colors"
              placeholder="Digite tokens separados por espaco..."
            />
          </div>

          {/* Seletor de estrategia */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">
              Estrategia de decodificacao
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {STRATEGY_OPTIONS.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setEstrategia(s.value)}
                  className={`px-3 py-2 rounded-sm border text-xs font-medium transition-all ${
                    estrategia === s.value
                      ? 'bg-cyan-50 border-cyan-200 text-cyan-700'
                      : 'bg-gray-100 border-gray-300 text-gray-600 hover:border-gray-400'
                  }`}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Slider
              value={temperatura}
              onChange={setTemperatura}
              min={0.1}
              max={2.0}
              step={0.1}
              label="Temperatura"
            />
            <Slider
              value={maxTokens}
              onChange={setMaxTokens}
              min={1}
              max={30}
              step={1}
              label="Max Tokens"
            />
          </div>

          {/* Botao de gerar */}
          <button
            onClick={handleGenerate}
            disabled={generate.loading || !prompt.trim()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-300 text-white text-sm font-medium transition-colors"
          >
            <Play size={14} weight="fill" /> Gerar Texto
          </button>

          {/* Token Stream */}
          <ApiLoadingState
            loading={generate.loading}
            error={generate.error}
            fallback={
              <TokenStream
                tokens={FALLBACK_TOKENS}
                autoPlay={true}
                speed={600}
              />
            }
          >
            {generate.data && (
              <div className="space-y-4">
                <TokenStream
                  tokens={generatedTokens}
                  probabilities={tokenProbs}
                  autoPlay={true}
                  speed={400}
                />

                {/* Texto completo gerado */}
                <div className="glass-card p-4">
                  <p className="text-xs font-medium text-gray-600 mb-1">Texto gerado:</p>
                  <p className="text-sm font-mono text-cyan-300">{generate.data.texto_gerado}</p>
                </div>

                {/* Historico de probabilidades */}
                {generate.data.historico_probabilidades.length > 0 && (
                  <div className="glass-card p-4 space-y-3">
                    <p className="text-xs font-medium text-gray-600">Top-5 tokens por posicao</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {generate.data.historico_probabilidades.map((h, i) => (
                        <div key={i} className="flex items-center gap-3">
                          <span className="text-xs font-mono text-cyan-400 w-16 flex-shrink-0 truncate">
                            {h.token}
                          </span>
                          <div className="flex-1 flex gap-1">
                            {h.top_5_tokens.map((t, j) => (
                              <div
                                key={j}
                                className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${
                                  t.token === h.token
                                    ? 'bg-cyan-50 border-cyan-200 text-cyan-700'
                                    : 'bg-gray-100 border-gray-200 text-gray-500'
                                }`}
                              >
                                {t.token} {(t.probabilidade * 100).toFixed(0)}%
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Info da estrategia */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 rounded-sm bg-gray-100">
                    <p className="text-xs text-gray-500">Estrategia</p>
                    <p className="text-sm font-mono text-cyan-400">{generate.data.estrategia}</p>
                  </div>
                  <div className="text-center p-3 rounded-sm bg-gray-100">
                    <p className="text-xs text-gray-500">Tokens gerados</p>
                    <p className="text-sm font-mono text-cyan-400">{generate.data.tokens_gerados.length}</p>
                  </div>
                </div>

                {/* Explicacao */}
                <p className="text-xs text-gray-500 leading-relaxed">
                  {generate.data.explicacao}
                </p>
              </div>
            )}
          </ApiLoadingState>
        </div>
      </EducationalViz>

      {/* Pipeline 3D de inferencia */}
      <EducationalViz
        title="Pipeline de Geracao 3D"
        icon={<Lightning size={18} weight="duotone" />}
        caption="Visualize o processo de geracao autorregressiva em 3D. Cada bloco representa um token gerado sequencialmente."
      >
        <InferencePipeline3D
          tokens={generatedTokens.length > 0 ? generatedTokens : FALLBACK_TOKENS}
          probabilities={tokenProbs && tokenProbs.length > 0 ? tokenProbs : undefined}
        />
      </EducationalViz>

      {/* Demonstracao de temperatura */}
      <EducationalViz
        title="Efeito da Temperatura"
        icon={<Lightning size={18} weight="duotone" />}
        caption="Compare como diferentes temperaturas afetam a distribuicao de probabilidade sobre os mesmos logits. T baixo = concentrado, T alto = espalhado."
        formula={{
          formula: "H = -\\sum_i p_i \\log(p_i)",
          variables: [
            { symbol: 'H', color: '#f59e0b', label: 'Entropia', description: 'Mede a incerteza - maior entropia = mais aleatorio' },
            { symbol: 'p_i', color: '#22d3ee', label: 'Probabilidade', description: 'Probabilidade ajustada de cada token' },
          ],
        }}
      >
        <ApiLoadingState
          loading={tempDemo.loading}
          error={tempDemo.error}
          fallback={
            <div className="text-sm text-gray-500 text-center py-8">
              Dados de demonstracao indisponiveis - inicie o backend
            </div>
          }
        >
          {tempDemo.data && (
            <div className="space-y-4">
              <PlotlyChart
                data={tempChartData}
                layout={{
                  title: { text: 'Distribuicao por Temperatura', font: { color: '#374151', size: 13 } },
                  barmode: 'group',
                  xaxis: { title: { text: 'Token', font: { color: '#6b7280' } } },
                  yaxis: { title: { text: 'Probabilidade', font: { color: '#6b7280' } } },
                  legend: { orientation: 'h', y: -0.2, font: { color: '#6b7280' } },
                  margin: { t: 40, r: 20, b: 60, l: 60 },
                }}
                height={350}
              />

              {/* Entropias */}
              <div className="flex flex-wrap gap-3 justify-center">
                {tempDemo.data.distribuicoes.map((dist) => (
                  <div key={dist.temperatura} className="text-center p-2 rounded-sm bg-gray-100 min-w-[90px]">
                    <p className="text-[10px] text-gray-500">T={dist.temperatura}</p>
                    <p className="text-xs font-mono text-amber-400">H={dist.entropia.toFixed(2)}</p>
                    <p className="text-[10px] text-gray-400 mt-0.5">{dist.descricao}</p>
                  </div>
                ))}
              </div>

              <p className="text-xs text-gray-500 leading-relaxed">
                {tempDemo.data.explicacao}
              </p>
            </div>
          )}
        </ApiLoadingState>
      </EducationalViz>

      {/* Comparacao de estrategias de amostragem */}
      <EducationalViz
        title="Comparacao de Estrategias"
        icon={<Lightning size={18} weight="duotone" />}
        caption="Cada estrategia gera texto diferente a partir do mesmo prompt. Compare os resultados lado a lado."
      >
        <ApiLoadingState
          loading={samplingDemo.loading}
          error={samplingDemo.error}
          fallback={
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {['Greedy', 'Temperatura Baixa', 'Temperatura Alta', 'Top-K', 'Top-P', 'Beam Search'].map((nome) => (
                <div key={nome} className="glass-card p-4">
                  <p className="text-xs font-semibold text-cyan-400 font-mono mb-1.5">{nome}</p>
                  <p className="text-xs text-gray-500">Dados indisponiveis offline</p>
                </div>
              ))}
            </div>
          }
        >
          {samplingDemo.data && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { key: 'greedy', label: 'Greedy', color: 'text-blue-400' },
                  { key: 'temperatura_baixa', label: 'Temperatura Baixa', color: 'text-green-400' },
                  { key: 'temperatura_alta', label: 'Temperatura Alta', color: 'text-red-400' },
                  { key: 'top_k', label: 'Top-K', color: 'text-purple-400' },
                  { key: 'top_p', label: 'Top-P (Nucleus)', color: 'text-amber-400' },
                ].map(({ key, label, color }) => {
                  const result = samplingDemo.data![key as keyof typeof samplingDemo.data] as Record<string, unknown> | undefined
                  if (!result || Array.isArray(result)) return null
                  return (
                    <div key={key} className="glass-card p-4">
                      <p className={`text-xs font-semibold font-mono mb-2 ${color}`}>{label}</p>
                      {result.texto_gerado != null && (
                        <p className="text-xs font-mono text-gray-700 mb-1">
                          &quot;{String(result.texto_gerado)}&quot;
                        </p>
                      )}
                      {result.tokens_gerados != null && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {(result.tokens_gerados as string[]).map((t, i) => (
                            <span key={i} className="px-1.5 py-0.5 bg-gray-100 border border-gray-200 rounded text-[10px] font-mono text-gray-600">
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}

                {/* Beam search (array) */}
                {samplingDemo.data.beam_search && samplingDemo.data.beam_search.length > 0 && (
                  <div className="glass-card p-4">
                    <p className="text-xs font-semibold font-mono mb-2 text-cyan-400">Beam Search</p>
                    {samplingDemo.data.beam_search.map((beam, i) => (
                      <div key={i} className="mb-1">
                        <p className="text-[10px] text-gray-500">Beam {i + 1}:</p>
                        <p className="text-xs font-mono text-gray-700">
                          "{String((beam as Record<string, unknown>).texto_gerado ?? '')}"
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <p className="text-xs text-gray-500 leading-relaxed">
                {samplingDemo.data.explicacao}
              </p>
            </div>
          )}
        </ApiLoadingState>
      </EducationalViz>

      {/* Passo a passo */}
      <StepByStep
        steps={inferenceSteps}
        title="Inferencia Passo a Passo"
        autoplaySpeed={4000}
      />

      {/* Por que importa */}
      <WhyItMatters>
        <p>
          A geracao autoregressiva e o mecanismo fundamental por tras de todos os LLMs modernos.
          Cada token gerado depende de todos os anteriores, criando uma cadeia de decisoes
          que produz texto coerente.
        </p>
        <p className="mt-2">
          A escolha da estrategia de decodificacao e crucial: greedy search e rapido mas repetitivo,
          enquanto nucleus sampling (top-p) produz texto mais diverso e natural. Na pratica,
          a maioria dos chatbots usa temperatura + top-p para equilibrar criatividade e coerencia.
        </p>
      </WhyItMatters>
    </div>
  )
}
