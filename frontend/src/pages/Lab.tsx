import { useState, useEffect } from 'react'
import { Flask, PaperPlaneTilt, SpinnerGap, Lightning, Eye, BracketsCurly, Waveform, ArrowCounterClockwise, Info } from '@phosphor-icons/react'
import {
  useTokenize,
  useGenerate,
  useSelfAttention,
  useEmbeddingSpace,
} from '../api/hooks'
import EmbeddingSpace from '../components/viz/EmbeddingSpace'
import TokenStream from '../components/viz/TokenStream'
import Heatmap3D from '../components/viz/Heatmap3D'
import ApiLoadingState from '../components/education/ApiLoadingState'

// ─── Laboratorio interativo ───────────────────────────────────────────────────

const CORES_TOKENS = [
  'bg-blue-50 text-blue-700 border-blue-200',
  'bg-violet-50 text-violet-700 border-violet-200',
  'bg-emerald-50 text-emerald-700 border-emerald-200',
  'bg-orange-50 text-orange-700 border-orange-200',
  'bg-pink-50 text-pink-700 border-pink-200',
  'bg-yellow-50 text-yellow-700 border-yellow-200',
]

type TabId = 'tokenizacao' | 'geracao' | 'embeddings' | 'atencao'

const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'tokenizacao', label: 'Tokenizacao', icon: <BracketsCurly size={14} weight="regular" /> },
  { id: 'geracao', label: 'Geracao', icon: <Lightning size={14} weight="duotone" /> },
  { id: 'embeddings', label: 'Embeddings', icon: <Eye size={14} weight="duotone" /> },
  { id: 'atencao', label: 'Atencao', icon: <Waveform size={14} weight="regular" /> },
]

const suggestedPrompts = [
  'o gato sentou no',
  'a casa era muito',
  'ele foi ate o',
  'nos temos uma grande',
  'a escola fica na',
]

const STRATEGY_OPTIONS = [
  {
    value: 'greedy',
    label: 'Greedy',
    desc: 'A cada passo, o modelo escolhe o token com a maior probabilidade - sem aleatoriedade. Dado o mesmo prompt, o resultado sera sempre identico. A vantagem e a previsibilidade, mas o texto tende a ficar repetitivo.',
  },
  {
    value: 'temperatura',
    label: 'Temperatura',
    desc: 'Antes de converter os logits em probabilidades, cada logit e dividido por T (temperatura). T < 1 = texto conservador. T > 1 = texto mais criativo. T = 1 mantem a distribuicao original.',
  },
  {
    value: 'top_k',
    label: 'Top-K',
    desc: 'O modelo ordena tokens por probabilidade e descarta todos exceto os K primeiros. As probabilidades dos K sobreviventes sao renormalizadas e um token e sorteado entre eles.',
  },
  {
    value: 'top_p',
    label: 'Top-P',
    desc: 'Tambem chamado de Nucleus Sampling. Ordena os tokens por probabilidade e vai somando ate atingir o limiar P (ex: 0.9 = 90%). E adaptativo: quando o modelo tem certeza, o nucleo tem poucos tokens; quando incerto, tem muitos.',
  },
]

const embeddingTestWords = [
  'gato', 'cachorro', 'peixe',
  'vermelho', 'azul', 'verde',
  'correr', 'andar', 'nadar',
  'casa', 'predio', 'escola',
]

const attentionTestTokens = ['O', 'gato', 'sentou', 'no', 'tapete', 'macio']

export default function Lab() {
  const [activeTab, setActiveTab] = useState<TabId>('tokenizacao')

  // ── Tokenization state ──
  const [textoToken, setTextoToken] = useState('Ola, como voce esta?')
  const tokenizeApi = useTokenize()

  useEffect(() => {
    if (textoToken.trim()) {
      tokenizeApi.execute({ texto: textoToken })
    }
  }, [textoToken]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Generation state ──
  const [prompt, setPrompt] = useState('')
  const [estrategia, setEstrategia] = useState('top_k')
  const [temperatura, setTemperatura] = useState(0.8)
  const [maxTokens, setMaxTokens] = useState(15)
  const generateApi = useGenerate()

  function handleGenerate() {
    if (!prompt.trim() || generateApi.loading) return
    generateApi.execute({
      prompt: prompt.trim().split(/\s+/),
      max_tokens: maxTokens,
      temperatura,
      estrategia,
      k: 40,
      p: 0.9,
    })
  }

  // ── Embeddings state ──
  const embeddingApi = useEmbeddingSpace()
  const [embeddingsLoaded, setEmbeddingsLoaded] = useState(false)

  useEffect(() => {
    if (activeTab === 'embeddings' && !embeddingsLoaded) {
      embeddingApi.execute({ tokens: embeddingTestWords, metodo: 'pca' })
      setEmbeddingsLoaded(true)
    }
  }, [activeTab]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (embeddingApi.error) {
      setEmbeddingsLoaded(false)
    }
  }, [embeddingApi.error])

  // ── Attention state ──
  const attentionApi = useSelfAttention()
  const [attentionLoaded, setAttentionLoaded] = useState(false)

  useEffect(() => {
    if (activeTab === 'atencao' && !attentionLoaded) {
      attentionApi.execute({ tokens: attentionTestTokens })
      setAttentionLoaded(true)
    }
  }, [activeTab]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (attentionApi.error) {
      setAttentionLoaded(false)
    }
  }, [attentionApi.error])

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Cabecalho */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Flask size={12} weight="duotone" />
          Modulo 10 - Laboratorio
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Laboratorio Interativo
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          Coloque em pratica tudo que voce aprendeu. Experimente a tokenizacao,
          explore embeddings, visualize atencao e interaja diretamente com o
          modelo via API.
        </p>
      </section>

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-sm border border-gray-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2.5 rounded-sm text-xs font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-teal-600 text-white shadow-lg'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-200'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'tokenizacao' && (
        <section className="glass-card p-6 space-y-4">
          <h3 className="text-sm font-semibold text-gray-800">
            Experimento: Tokenizacao via API
          </h3>
          <p className="text-xs text-gray-500">
            Digite texto e veja como o tokenizador BPE o divide em tokens.
          </p>
          <textarea
            value={textoToken}
            readOnly
            rows={2}
            className="w-full bg-gray-50 border border-gray-200 rounded-sm px-4 py-2.5 font-mono text-sm resize-none cursor-default"
          />

          <ApiLoadingState
            loading={tokenizeApi.loading}
            error={tokenizeApi.error}
            compact
            fallback={
              <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                {(textoToken.match(/[\w']+|[^\s\w]/g) ?? []).map(
                  (t, i) => (
                    <span
                      key={i}
                      className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-mono ${
                        CORES_TOKENS[i % CORES_TOKENS.length]
                      }`}
                    >
                      {t}
                    </span>
                  )
                )}
              </div>
            }
          >
            {tokenizeApi.data && (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-1.5 min-h-[32px]">
                  {tokenizeApi.data.tokens.map(
                    (t: string, i: number) => (
                      <span
                        key={i}
                        className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-mono ${
                          CORES_TOKENS[i % CORES_TOKENS.length]
                        }`}
                      >
                        {t}
                      </span>
                    )
                  )}
                </div>
                <div className="flex gap-4 text-xs text-gray-500">
                  <span>
                    {tokenizeApi.data.tokens.length} token
                    {tokenizeApi.data.tokens.length !== 1 ? 's' : ''}
                  </span>
                  {tokenizeApi.data.num_tokens != null && (
                    <span>Total: {tokenizeApi.data.num_tokens}</span>
                  )}
                </div>
              </div>
            )}
          </ApiLoadingState>
        </section>
      )}

      {activeTab === 'geracao' && (
        <section className="glass-card p-6 space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-gray-800">
              Experimento: Geracao de texto
            </h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Selecione um prompt, escolha a estrategia de decodificacao e observe o modelo gerar tokens um a um.
            </p>
          </div>

          {/* Banner simulacao */}
          <div className="flex items-start gap-2.5 bg-amber-50 border border-amber-200 rounded-sm px-4 py-3">
            <Info size={16} weight="duotone" className="text-amber-600 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-amber-800 leading-relaxed">
              Este e um simulador didatico com vocabulario limitado (~50 palavras). As probabilidades
              sao calculadas com pesos aleatorios para demonstrar o mecanismo de geracao - nao refletem
              um modelo treinado real.
            </p>
          </div>

          {/* Prompt selector */}
          <div>
            <label className="text-xs font-medium text-gray-600 mb-2 block">Selecione um prompt</label>
            <div className="flex flex-wrap gap-2">
              {suggestedPrompts.map((s) => (
                <button
                  key={s}
                  onClick={() => setPrompt(s)}
                  className={`px-3 py-1.5 rounded-sm border text-xs font-mono transition-all ${
                    prompt === s
                      ? 'bg-teal-50 border-teal-300 text-teal-700'
                      : 'bg-gray-100 hover:bg-gray-200 border-gray-200 text-gray-600 hover:border-gray-300'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Strategy selector */}
          <div>
            <label className="text-xs font-medium text-gray-600 mb-2 block">Estrategia de decodificacao</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {STRATEGY_OPTIONS.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setEstrategia(s.value)}
                  className={`text-left px-3 py-2 rounded-sm border transition-all ${
                    estrategia === s.value
                      ? 'bg-teal-50 border-teal-300'
                      : 'bg-white border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <span className={`text-xs font-medium block ${
                    estrategia === s.value ? 'text-teal-700' : 'text-gray-700'
                  }`}>
                    {s.label}
                  </span>
                  <span className="text-[10px] text-gray-500 leading-tight block mt-0.5">
                    {s.desc.split('.')[0]}.
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-gray-600 mb-1 block">
                Temperatura: {temperatura.toFixed(1)}
              </label>
              <input
                type="range"
                min={0.1}
                max={2.0}
                step={0.1}
                value={temperatura}
                onChange={(e) => setTemperatura(parseFloat(e.target.value))}
                className="w-full accent-teal-600"
              />
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>0.1 (conservador)</span>
                <span>2.0 (criativo)</span>
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 mb-1 block">
                Max tokens: {maxTokens}
              </label>
              <input
                type="range"
                min={5}
                max={30}
                step={1}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                className="w-full accent-teal-600"
              />
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>5</span>
                <span>30</span>
              </div>
            </div>
          </div>

          {/* Generate button */}
          <button
            onClick={handleGenerate}
            disabled={!prompt.trim() || generateApi.loading}
            className="btn-primary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {generateApi.loading ? (
              <>
                <SpinnerGap size={14} weight="bold" className="animate-spin" />
                Gerando...
              </>
            ) : (
              <>
                <PaperPlaneTilt size={14} weight="fill" />
                Gerar texto
              </>
            )}
          </button>

          {/* Results */}
          <ApiLoadingState
            loading={generateApi.loading}
            error={generateApi.error}
            compact
            fallback={
              <div className="space-y-3">
                <TokenStream
                  tokens={['o', 'gato', 'sentou', 'no', 'tapete']}
                  autoPlay={true}
                  speed={600}
                />
                <p className="text-[10px] text-gray-400 text-center">
                  Dados de exemplo - selecione um prompt e clique em gerar
                </p>
              </div>
            }
          >
            {generateApi.data && (
              <div className="space-y-4">
                {/* Token Stream */}
                <TokenStream
                  tokens={generateApi.data.tokens_gerados ?? []}
                  probabilities={generateApi.data.historico_probabilidades?.map(
                    (h: { top_5_tokens: { probabilidade: number }[] }) => h.top_5_tokens.map((t: { probabilidade: number }) => t.probabilidade)
                  )}
                  autoPlay={true}
                  speed={400}
                />

                {/* Top-5 candidatos por passo */}
                {generateApi.data.historico_probabilidades && generateApi.data.historico_probabilidades.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-gray-600">Top-5 candidatos por passo</p>
                    <div className="space-y-1.5 max-h-56 overflow-y-auto">
                      {generateApi.data.historico_probabilidades.map((h: { token: string; top_5_tokens: { token: string; probabilidade: number }[] }, i: number) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-[11px] font-mono text-teal-600 w-14 flex-shrink-0 truncate font-medium">
                            {h.token}
                          </span>
                          <div className="flex-1 flex gap-1">
                            {h.top_5_tokens.map((t: { token: string; probabilidade: number }, j: number) => (
                              <div key={j} className="flex-1 max-w-[120px]">
                                <div className="flex items-center gap-1">
                                  <span className={`text-[10px] font-mono ${
                                    t.token === h.token ? 'text-teal-700 font-medium' : 'text-gray-500'
                                  }`}>
                                    {t.token}
                                  </span>
                                  <span className="text-[9px] text-gray-400">
                                    {(t.probabilidade * 100).toFixed(0)}%
                                  </span>
                                </div>
                                <div className="h-1 bg-gray-100 rounded-full mt-0.5">
                                  <div
                                    className={`h-1 rounded-full ${
                                      t.token === h.token ? 'bg-teal-500' : 'bg-gray-300'
                                    }`}
                                    style={{ width: `${Math.max(t.probabilidade * 100, 2)}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Summary row */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="text-center p-3 rounded-sm bg-gray-50 border border-gray-200">
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">Estrategia</p>
                    <p className="text-sm font-mono text-teal-600 mt-1">{generateApi.data.estrategia}</p>
                  </div>
                  <div className="text-center p-3 rounded-sm bg-gray-50 border border-gray-200">
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">Tokens gerados</p>
                    <p className="text-sm font-mono text-teal-600 mt-1">{generateApi.data.tokens_gerados?.length ?? 0}</p>
                  </div>
                  <div className="text-center p-3 rounded-sm bg-gray-50 border border-gray-200">
                    <p className="text-[10px] text-gray-500 uppercase tracking-wider">Temperatura</p>
                    <p className="text-sm font-mono text-teal-600 mt-1">{temperatura.toFixed(1)}</p>
                  </div>
                </div>

                {/* Explicacao */}
                {generateApi.data.explicacao && (
                  <p className="text-xs text-gray-500 leading-relaxed">
                    {generateApi.data.explicacao}
                  </p>
                )}
              </div>
            )}
          </ApiLoadingState>
        </section>
      )}

      {activeTab === 'embeddings' && (
        <section className="glass-card p-6 space-y-4">
          <h3 className="text-sm font-semibold text-gray-800">
            Experimento: Espaco de Embeddings 3D
          </h3>
          <p className="text-xs text-gray-500">
            Visualize como palavras similares ficam proximas no espaco vetorial.
            Palavras da mesma categoria semantica tendem a formar clusters.
          </p>

          <ApiLoadingState
            loading={embeddingApi.loading}
            error={embeddingApi.error}
            compact
            loadingMessage="Calculando embeddings e reduzindo dimensionalidade..."
            fallback={
              <div className="py-8 text-center text-xs text-gray-500">
                Inicie o backend para visualizar embeddings em 3D.
              </div>
            }
          >
            {embeddingApi.data && embeddingApi.data.pontos && (
              <EmbeddingSpace
                points={embeddingApi.data.pontos}
                height={450}
              />
            )}
          </ApiLoadingState>

          {embeddingApi.error && (
            <button
              onClick={() => {
                setEmbeddingsLoaded(false)
                embeddingApi.execute({ tokens: embeddingTestWords, metodo: 'pca' })
                setEmbeddingsLoaded(true)
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              <ArrowCounterClockwise size={14} weight="regular" />
              Tentar novamente
            </button>
          )}

          <div className="flex flex-wrap gap-1.5">
            {embeddingTestWords.map((w) => (
              <span
                key={w}
                className="px-2 py-0.5 rounded text-[11px] font-mono bg-gray-100 border border-gray-200 text-gray-600"
              >
                {w}
              </span>
            ))}
          </div>
        </section>
      )}

      {activeTab === 'atencao' && (
        <section className="glass-card p-6 space-y-4">
          <h3 className="text-sm font-semibold text-gray-800">
            Experimento: Mapa de Atencao
          </h3>
          <p className="text-xs text-gray-500">
            Visualize os pesos de self-attention entre tokens. Cores mais
            intensas indicam que o modelo "presta mais atencao" entre aquele par
            de tokens.
          </p>

          <ApiLoadingState
            loading={attentionApi.loading}
            error={attentionApi.error}
            compact
            loadingMessage="Calculando pesos de atencao..."
            fallback={
              <div className="py-8 text-center text-xs text-gray-500">
                Inicie o backend para visualizar o mapa de atencao.
              </div>
            }
          >
            {attentionApi.data && attentionApi.data.pesos_atencao && (
              <Heatmap3D
                matrix={attentionApi.data.pesos_atencao.slice(0, attentionApi.data.num_tokens_reais).map(row => row.slice(0, attentionApi.data!.num_tokens_reais))}
                xLabels={attentionTestTokens}
                yLabels={attentionTestTokens}
                title="Self-Attention Weights"
                colorscale="YlOrRd"
                height={400}
                mode="2d"
                showValues
              />
            )}
          </ApiLoadingState>

          {attentionApi.error && (
            <button
              onClick={() => {
                setAttentionLoaded(false)
                attentionApi.execute({ tokens: attentionTestTokens })
                setAttentionLoaded(true)
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              <ArrowCounterClockwise size={14} weight="regular" />
              Tentar novamente
            </button>
          )}

          <div className="flex flex-wrap gap-1.5">
            {attentionTestTokens.map((t, i) => (
              <span
                key={i}
                className="px-2 py-0.5 rounded text-[11px] font-mono bg-query/10 border border-query/30 text-query"
              >
                {t}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
