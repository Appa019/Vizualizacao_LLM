import { useState, useEffect } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Flask, PaperPlaneTilt, SpinnerGap, Sparkle, Eye, BracketsCurly, Waveform, ArrowCounterClockwise } from '@phosphor-icons/react'
import {
  useTokenize,
  useGenerate,
  useAvailableModels,
  useSelfAttention,
  useEmbeddingSpace,
  useRealAttention,
  useSetupStatus,
} from '../api/hooks'
import type { LayoutContext } from '../components/layout/Layout'
import EmbeddingSpace from '../components/viz/EmbeddingSpace'
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
  { id: 'geracao', label: 'Geracao', icon: <Sparkle size={14} weight="duotone" /> },
  { id: 'embeddings', label: 'Embeddings', icon: <Eye size={14} weight="duotone" /> },
  { id: 'atencao', label: 'Atencao', icon: <Waveform size={14} weight="regular" /> },
]

const suggestedPrompts = [
  'Explique o que e um token em uma frase.',
  'Qual e a diferenca entre pre-treino e fine-tuning?',
  'Descreva o mecanismo de atencao com uma analogia.',
  'O que sao embeddings e por que sao importantes?',
  'Como funciona a geracao de texto em um LLM?',
]

const embeddingTestWords = [
  'gato', 'cachorro', 'peixe',
  'vermelho', 'azul', 'verde',
  'correr', 'andar', 'nadar',
  'casa', 'predio', 'escola',
]

const attentionTestTokens = ['O', 'gato', 'sentou', 'no', 'tapete', 'macio']

export default function Lab() {
  const { modoSimulacao } = useOutletContext<LayoutContext>()
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
  const generateApi = useGenerate()

  function handleGenerate() {
    if (!prompt.trim() || generateApi.loading) return
    generateApi.execute({
      prompt: [prompt.trim()],
      max_tokens: 150,
      temperatura: 0.7,
      estrategia: 'top_k',
      k: 50,
    })
  }

  // ── Models ──
  const modelsApi = useAvailableModels()

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
  const realAttnApi = useRealAttention()
  const { data: setupStatus } = useSetupStatus()

  const modeloCarregado = setupStatus?.modelo_carregado === true
  const modeloNome = setupStatus?.modelo_nome ?? ''
  const usandoReal = !modoSimulacao && modeloCarregado

  useEffect(() => {
    if (activeTab === 'atencao' && !attentionLoaded) {
      if (usandoReal) {
        realAttnApi.execute({ nome_modelo: modeloNome, texto: attentionTestTokens.join(' ') })
      } else {
        attentionApi.execute({ tokens: attentionTestTokens })
      }
      setAttentionLoaded(true)
    }
  }, [activeTab, usandoReal]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (attentionApi.error || realAttnApi.error) {
      setAttentionLoaded(false)
    }
  }, [attentionApi.error, realAttnApi.error])

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Cabecalho */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Flask size={12} weight="duotone" />
          Modulo 10 -- Laboratorio
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

      {/* Model status */}
      <section className="glass-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-xs text-gray-500 bg-gray-100 px-2.5 py-1 rounded border border-gray-300">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse" />
              Backend API
            </span>
            {modelsApi.data && (
              <span className="text-xs text-gray-500">
                {modelsApi.data.modelos?.length ?? 0} modelos
                disponiveis
              </span>
            )}
          </div>
          <a
            href="/setup"
            className="text-xs text-teal-400 hover:text-teal-300 transition-colors"
          >
            Configurar modelos →
          </a>
        </div>
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
            onChange={(e) => setTextoToken(e.target.value)}
            rows={2}
            placeholder="Digite para tokenizar..."
            className="w-full input-dark font-mono text-sm resize-none"
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
                  {tokenizeApi.data.contagem != null && (
                    <span>IDs: {tokenizeApi.data.contagem}</span>
                  )}
                </div>
              </div>
            )}
          </ApiLoadingState>
        </section>
      )}

      {activeTab === 'geracao' && (
        <section className="glass-card p-6 space-y-4">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-800">
                Experimento: Geracao de texto
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                Envie um prompt e veja o modelo gerar texto token por token.
              </p>
            </div>
          </div>

          {/* Suggested prompts */}
          <div className="flex flex-wrap gap-2">
            {suggestedPrompts.map((s) => (
              <button
                key={s}
                onClick={() => setPrompt(s)}
                className="px-2.5 py-1 rounded-sm bg-gray-100 hover:bg-gray-200 border border-gray-200 hover:border-teal-300 text-[11px] text-gray-600 hover:text-teal-600 transition-all"
              >
                {s}
              </button>
            ))}
          </div>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleGenerate()
            }}
            rows={3}
            placeholder="Digite seu prompt aqui... (Ctrl+Enter para enviar)"
            className="w-full input-dark font-mono text-sm resize-none"
          />

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
                Enviar prompt
              </>
            )}
          </button>

          <ApiLoadingState
            loading={generateApi.loading}
            error={generateApi.error}
            compact
            fallback={
              <div>
                <p className="text-xs text-gray-500 mb-2">Resposta do modelo (exemplo):</p>
                <div className="bg-white rounded-sm p-4 border border-gray-200">
                  <p className="text-sm text-gray-800 leading-relaxed font-mono whitespace-pre-wrap">
                    Um token e a menor unidade de texto que um modelo de linguagem processa. Pode ser uma palavra inteira, parte de uma palavra ou ate um unico caractere. Por exemplo, a palavra "tokenizacao" pode ser dividida em "token", "iza" e "cao".
                  </p>
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  32 tokens gerados
                </p>
              </div>
            }
          >
            {generateApi.data && (
              <div>
                <p className="text-xs text-gray-500 mb-2">Resposta do modelo:</p>
                <div className="bg-white rounded-sm p-4 border border-gray-200">
                  <p className="text-sm text-gray-800 leading-relaxed font-mono whitespace-pre-wrap">
                    {generateApi.data.texto_gerado ??
                      generateApi.data.tokens_gerados?.join('') ??
                      JSON.stringify(generateApi.data)}
                  </p>
                </div>
                {generateApi.data.tokens_gerados && (
                  <p className="text-xs text-gray-400 mt-2">
                    {generateApi.data.tokens_gerados.length} tokens gerados
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
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-800">
              Experimento: Mapa de Atencao
            </h3>
            <span className="flex items-center gap-1.5 text-[11px] text-gray-500 bg-gray-100 px-2 py-0.5 rounded-sm border border-gray-200">
              <span className={`w-1.5 h-1.5 rounded-full ${usandoReal ? 'bg-green-400' : 'bg-blue-400'}`} />
              {usandoReal ? `Modelo Real - ${modeloNome}` : 'Simulacao'}
            </span>
          </div>
          <p className="text-xs text-gray-500">
            Visualize os pesos de self-attention entre tokens. Cores mais
            intensas indicam que o modelo "presta mais atencao" entre aquele par
            de tokens.
          </p>

          {usandoReal ? (
            <ApiLoadingState
              loading={realAttnApi.loading}
              error={realAttnApi.error}
              compact
              loadingMessage="Obtendo pesos de atencao do modelo real..."
              fallback={
                <div className="py-8 text-center text-xs text-gray-500">
                  Inicie o backend para visualizar o mapa de atencao.
                </div>
              }
            >
              {realAttnApi.data && realAttnApi.data.camadas?.[0] && (
                <Heatmap3D
                  matrix={realAttnApi.data.camadas[0].media_cabecas}
                  xLabels={realAttnApi.data.tokens}
                  yLabels={realAttnApi.data.tokens}
                  title={`Atencao Real - Camada 1 (media das cabecas)`}
                  colorscale="YlOrRd"
                  height={400}
                  mode="2d"
                  showValues
                />
              )}
            </ApiLoadingState>
          ) : (
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
          )}

          {(attentionApi.error || realAttnApi.error) && (
            <button
              onClick={() => {
                setAttentionLoaded(false)
                if (usandoReal) {
                  realAttnApi.execute({ nome_modelo: modeloNome, texto: attentionTestTokens.join(' ') })
                } else {
                  attentionApi.execute({ tokens: attentionTestTokens })
                }
                setAttentionLoaded(true)
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              <ArrowCounterClockwise size={14} weight="regular" />
              Tentar novamente
            </button>
          )}

          <div className="flex flex-wrap gap-1.5">
            {(usandoReal && realAttnApi.data ? realAttnApi.data.tokens : attentionTestTokens).map((t, i) => (
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
