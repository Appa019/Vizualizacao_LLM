import { useState, useEffect } from 'react'
import { useOutletContext, Link } from 'react-router-dom'
import { Eye, Warning } from '@phosphor-icons/react'
import {
  useSelfAttention,
  useMultiHeadAttention,
  useAttentionFlow,
  useTokenImportance,
  useTokenize,
  useRealAttention,
  useSetupStatus,
} from '../api/hooks'
import type { LayoutContext } from '../components/layout/Layout'
import Heatmap3D from '../components/viz/Heatmap3D'
import AttentionFlow from '../components/viz/AttentionFlow'
import StepByStep from '../components/education/StepByStep'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import Slider from '../components/ui/Slider'
import Toggle from '../components/ui/Toggle'
import ApiLoadingState from '../components/education/ApiLoadingState'
import PlotlyChart from '../components/viz/PlotlyChart'
import AttentionWeights3D from '../components/viz/AttentionWeights3D'

// ─── Dados fallback quando o backend esta offline ────────────────────────────

const FALLBACK_TOKENS = ['O', 'gato', 'pulou', 'no', 'jardim']
const FALLBACK_PESOS: number[][] = [
  [0.85, 0.05, 0.04, 0.03, 0.03],
  [0.12, 0.62, 0.14, 0.06, 0.06],
  [0.06, 0.25, 0.48, 0.12, 0.09],
  [0.04, 0.08, 0.15, 0.65, 0.08],
  [0.03, 0.10, 0.18, 0.22, 0.47],
]

// ─── Variaveis das formulas ─────────────────────────────────────────────────

const attentionVars = [
  { symbol: 'Q', color: '#ef4444', label: 'Query', description: '"O que estou procurando?" - cada token faz uma pergunta ao contexto' },
  { symbol: 'K', color: '#22c55e', label: 'Key', description: '"O que tenho para oferecer?" - cada token expoe o que sabe' },
  { symbol: 'V', color: '#3b82f6', label: 'Value', description: '"Qual e minha informacao?" - o conteudo real transferido' },
  { symbol: 'd_k', color: '#fb923c', label: 'Dimensao da chave', description: 'Dimensao dos vetores Q e K - usado para escalonar os scores' },
]

// ─── Componente principal ───────────────────────────────────────────────────

export default function Attention() {
  const { modoSimulacao } = useOutletContext<LayoutContext>()
  const [inputText, setInputText] = useState('O gato pulou no jardim')
  const [tokens, setTokens] = useState<string[]>(FALLBACK_TOKENS)
  const [tokenSelecionado, setTokenSelecionado] = useState(0)
  const [modo3D, setModo3D] = useState(false)
  const [numCabecas, setNumCabecas] = useState(4)
  const [camadaSelecionada, setCamadaSelecionada] = useState(1)
  const [totalCamadasReal, setTotalCamadasReal] = useState(1)

  const tokenizer = useTokenize()
  const selfAttn = useSelfAttention()
  const multiHead = useMultiHeadAttention()
  const flow = useAttentionFlow()
  const importance = useTokenImportance()
  const realAttn = useRealAttention()
  const { data: setupStatus } = useSetupStatus()

  const modeloCarregado = setupStatus?.modelo_carregado === true
  const modeloNome = setupStatus?.modelo_nome ?? ''
  const usandoReal = !modoSimulacao && modeloCarregado

  // Tokenizar e buscar dados ao mudar o texto
  useEffect(() => {
    tokenizer.execute({ texto: inputText })
  }, [inputText])

  useEffect(() => {
    if (tokenizer.data) {
      const tks = tokenizer.data.tokens
      setTokens(tks)
      if (tokenSelecionado >= tks.length) setTokenSelecionado(0)
      selfAttn.execute({ tokens: tks, d_model: 64 })
      multiHead.execute({ tokens: tks, d_model: 64, num_cabecas: numCabecas })
      flow.execute({ tokens: tks, indice_token: tokenSelecionado, d_model: 64 })
      importance.execute({ tokens: tks, d_model: 64 })
    }
  }, [tokenizer.data])

  // Fetch real attention when in real mode or text changes
  useEffect(() => {
    if (usandoReal && inputText.trim()) {
      realAttn.execute({ nome_modelo: modeloNome, texto: inputText, camada: camadaSelecionada })
    }
  }, [usandoReal, inputText, modeloNome])

  // Track total layers from model info
  useEffect(() => {
    if (setupStatus?.modelo_info) {
      setTotalCamadasReal(setupStatus.modelo_info.num_camadas)
    }
  }, [setupStatus])

  // Re-fetch real attention when layer changes
  useEffect(() => {
    if (usandoReal && inputText.trim()) {
      realAttn.execute({ nome_modelo: modeloNome, texto: inputText, camada: camadaSelecionada })
    }
  }, [camadaSelecionada])

  // Re-fetch flow ao mudar token selecionado
  useEffect(() => {
    if (tokens.length > 0) {
      flow.execute({ tokens, indice_token: tokenSelecionado, d_model: 64 })
    }
  }, [tokenSelecionado])

  // Re-fetch multi-head ao mudar numero de cabecas
  useEffect(() => {
    if (tokens.length > 0) {
      multiHead.execute({ tokens, d_model: 64, num_cabecas: numCabecas })
    }
  }, [numCabecas])

  // Dados para exibicao - switch between simulation and real
  const realData = realAttn.data
  const realCamada = realData?.camadas?.[0]

  const attnData = selfAttn.data
  const displayTokens = usandoReal && realData
    ? realData.tokens
    : attnData
      ? attnData.tokens.slice(0, attnData.num_tokens_reais)
      : tokens
  const displayPesos = usandoReal && realCamada
    ? realCamada.media_cabecas
    : attnData
      ? attnData.pesos_atencao.slice(0, attnData.num_tokens_reais).map((r) => r.slice(0, attnData.num_tokens_reais))
      : FALLBACK_PESOS

  // StepByStep passos
  const steps = attnData
    ? attnData.passos_explicados.map((p, i) => ({
        title: p.passo,
        description: p.descricao,
        content: (
          <div className="text-xs font-mono text-gray-500 bg-white rounded-sm p-3 overflow-x-auto">
            {i === 0 && `Q shape: [${attnData.num_tokens_reais}, 64] | K shape: [${attnData.num_tokens_reais}, 64] | V shape: [${attnData.num_tokens_reais}, 64]`}
            {i === 1 && `Scores shape: [${attnData.num_tokens_reais}, ${attnData.num_tokens_reais}] - cada token tem um score para cada outro token`}
            {i === 2 && `Fator de escala: 1/${'\u221A'}${64} = ${attnData.fator_escala.toFixed(4)}`}
            {i === 3 && `Cada linha da matriz soma 1.0 - distribuicao de probabilidade`}
            {i === 4 && `Saida shape: [${attnData.num_tokens_reais}, 64] - mesma dimensao dos embeddings`}
          </div>
        ),
      }))
    : []

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Header */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Eye size={12} weight="duotone" />
          Modulo 06
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Mecanismo de Atencao
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          A atencao permite que cada token "observe" todos os outros tokens na sequencia
          e decida quanto peso dar a cada um. E o que da ao Transformer sua capacidade
          de capturar dependencias de longo alcance.
        </p>
      </section>

      {/* Warning banner when real model selected but not loaded */}
      {!modoSimulacao && !modeloCarregado && (
        <section className="flex items-center gap-3 p-4 rounded-sm bg-amber-50 border border-amber-200">
          <Warning size={18} weight="fill" className="text-amber-500 flex-shrink-0" />
          <p className="text-sm text-amber-700">
            Modo "Modelo Real" selecionado, mas nenhum modelo esta carregado.{' '}
            <Link to="/setup" className="underline font-medium hover:text-amber-800">
              Configure um modelo no Setup
            </Link>{' '}
            ou volte para "Simulacao".
          </p>
        </section>
      )}

      {/* Data source label */}
      {usandoReal && (
        <section className="flex items-center gap-2 text-xs text-gray-500">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
          Modelo Real - {modeloNome}
        </section>
      )}

      {/* Input de texto */}
      <section className="glass-card p-5">
        <label className="text-sm font-medium text-gray-700 mb-2 block">
          Texto de entrada
        </label>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="w-full bg-white border border-gray-300 rounded-sm px-4 py-2.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-blue-500/50 transition-colors"
          placeholder="Digite uma frase para analisar..."
        />
        <div className="flex flex-wrap gap-1.5 mt-3">
          {displayTokens.map((token, i) => (
            <button
              key={i}
              onClick={() => setTokenSelecionado(i)}
              className={`px-2.5 py-1 rounded-sm border text-xs font-mono font-medium transition-all ${
                tokenSelecionado === i
                  ? 'bg-blue-50 border-blue-200 text-blue-700 scale-105'
                  : 'bg-gray-100 border-gray-300 text-gray-700 hover:border-gray-400'
              }`}
            >
              {token}
            </button>
          ))}
        </div>

        {/* Layer selector for real model */}
        {usandoReal && totalCamadasReal > 1 && (
          <div className="mt-4">
            <Slider
              value={camadaSelecionada}
              onChange={setCamadaSelecionada}
              min={1}
              max={totalCamadasReal}
              step={1}
              label="Camada do modelo"
              unit={`/ ${totalCamadasReal}`}
            />
          </div>
        )}
      </section>

      {/* StepByStep */}
      {steps.length > 0 && (
        <StepByStep
          steps={steps}
          title="Self-Attention Passo a Passo"
          autoplaySpeed={4000}
        />
      )}

      {/* Formula KaTeX */}
      <section>
        <FormulaBlock
          formula="\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V"
          variables={attentionVars}
          size="lg"
        />
      </section>

      {/* Heatmap de atencao */}
      <EducationalViz
        title="Mapa de Atencao"
        icon={<Eye size={18} weight="duotone" />}
        caption="Cada celula mostra quanto o token da linha presta atencao ao token da coluna. Cores mais intensas = mais atencao."
        formula={{
          formula: "w_{ij} = \\text{softmax}\\left(\\frac{q_i \\cdot k_j}{\\sqrt{d_k}}\\right)",
          variables: [
            { symbol: 'w_{ij}', color: '#60a5fa', label: 'Peso de atencao', description: 'Quanto o token i presta atencao ao token j' },
            { symbol: 'q_i', color: '#ef4444', label: 'Query do token i', description: 'Vetor query do token atual' },
            { symbol: 'k_j', color: '#22c55e', label: 'Key do token j', description: 'Vetor key do token alvo' },
          ],
        }}
        whyItMatters={
          <span>
            A atencao permite que o modelo entenda que em "O gato que o cachorro perseguiu fugiu",
            "fugiu" se refere a "gato" e nao a "cachorro" - mesmo com tres palavras entre eles.
            Sem atencao, modelos antigos (RNNs) tinham dificuldade com essas dependencias de longa distancia.
          </span>
        }
      >
        <div className="space-y-3">
          <div className="flex items-center justify-end gap-3">
            <Toggle
              enabled={modo3D}
              onChange={setModo3D}
              labelLeft="2D"
              labelRight="3D"
              size="sm"
            />
          </div>
          <ApiLoadingState
            loading={selfAttn.loading}
            error={selfAttn.error}
            fallback={
              <Heatmap3D
                matrix={FALLBACK_PESOS}
                xLabels={FALLBACK_TOKENS}
                yLabels={FALLBACK_TOKENS}
                mode="2d"
              />
            }
          >
            <Heatmap3D
              matrix={displayPesos}
              xLabels={displayTokens}
              yLabels={displayTokens}
              mode={modo3D ? '3d' : '2d'}
              height={modo3D ? 500 : 400}
              colorscale="YlOrRd"
            />
          </ApiLoadingState>
        </div>
      </EducationalViz>

      {/* Fluxo de atencao */}
      <EducationalViz
        title="Fluxo de Atencao"
        icon={<Eye size={18} weight="duotone" />}
        caption={`Mostra como o token "${displayTokens[tokenSelecionado] || '...'}" distribui sua atencao pela sequencia. Linhas mais grossas = mais atencao.`}
      >
        <ApiLoadingState
          loading={flow.loading}
          error={flow.error}
          fallback={
            <AttentionFlow
              tokens={FALLBACK_TOKENS}
              weights={FALLBACK_PESOS[0]}
              sourceToken={0}
            />
          }
        >
          <AttentionFlow
            tokens={displayTokens}
            weights={flow.data?.pesos?.slice(0, displayTokens.length) ?? displayPesos[tokenSelecionado] ?? []}
            sourceToken={tokenSelecionado}
          />
        </ApiLoadingState>
      </EducationalViz>

      {/* Atencao 3D interativa */}
      <EducationalViz
        title="Pesos de Atencao em 3D"
        icon={<Eye size={18} weight="duotone" />}
        caption="Tokens dispostos em circulo com arcos representando a forca da atencao. Passe o mouse para explorar as conexoes."
      >
        <AttentionWeights3D
          tokens={displayTokens}
          weights={displayPesos}
        />
      </EducationalViz>

      {/* Multi-Head Attention */}
      <EducationalViz
        title="Multi-Head Attention"
        icon={<Eye size={18} weight="duotone" />}
        caption={`${numCabecas} cabecas operando em paralelo - cada uma aprende a focar em diferentes tipos de relacoes.`}
        formula={{
          formula: "\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W^O",
          variables: [
            { symbol: 'h', color: '#a855f7', label: 'Num. cabecas', description: `Numero de cabecas de atencao (${numCabecas})` },
            { symbol: 'W^O', color: '#f59e0b', label: 'Projecao de saida', description: 'Matriz que projeta a concatenacao de volta a d_model' },
          ],
        }}
      >
        <div className="space-y-4">
          <Slider
            value={numCabecas}
            onChange={(v) => {
              // Only allow divisors of 64
              const divisors = [1, 2, 4, 8, 16]
              const nearest = divisors.reduce((prev, curr) =>
                Math.abs(curr - v) < Math.abs(prev - v) ? curr : prev
              )
              setNumCabecas(nearest)
            }}
            min={1}
            max={16}
            step={1}
            label="Numero de cabecas"
            unit=""
          />

          <ApiLoadingState loading={multiHead.loading} error={multiHead.error}>
            {multiHead.data && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {multiHead.data.cabecas.map((cabeca) => (
                  <div key={cabeca.cabeca} className="space-y-1">
                    <p className="text-xs font-medium text-gray-600 text-center">
                      Cabeca {cabeca.cabeca}
                    </p>
                    <Heatmap3D
                      matrix={cabeca.pesos_atencao}
                      xLabels={cabeca.tokens}
                      yLabels={cabeca.tokens}
                      height={180}
                      showValues={false}
                      colorscale="Blues"
                    />
                  </div>
                ))}
              </div>
            )}
          </ApiLoadingState>

          {multiHead.data && (
            <p className="text-xs text-gray-500 leading-relaxed">
              {multiHead.data.explicacao}
            </p>
          )}
        </div>
      </EducationalViz>

      {/* Importancia dos tokens */}
      <EducationalViz
        title="Importancia dos Tokens"
        icon={<Eye size={18} weight="duotone" />}
        caption="Metricas de quanto cada token e importante na sequencia - baseado nos pesos de atencao."
      >
        <ApiLoadingState
          loading={importance.loading}
          error={importance.error}
        >
          {importance.data && (
            <div className="space-y-4">
              <PlotlyChart
                data={[
                  {
                    type: 'bar',
                    x: importance.data.importancia_combinada,
                    y: importance.data.tokens,
                    orientation: 'h',
                    marker: {
                      color: importance.data.importancia_combinada.map((v, i) =>
                        importance.data!.tokens[i] === importance.data!.token_mais_importante
                          ? '#ef4444'
                          : '#3b82f6'
                      ),
                    },
                    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
                  } as Plotly.Data,
                ]}
                layout={{
                  title: { text: 'Importancia Combinada', font: { color: '#374151', size: 13 } },
                  yaxis: { autorange: 'reversed' as const },
                  margin: { l: 80, r: 20, t: 40, b: 30 },
                }}
                height={Math.max(200, importance.data.tokens.length * 35)}
              />
              <p className="text-xs text-gray-500">
                {importance.data.explicacao}
              </p>
            </div>
          )}
        </ApiLoadingState>
      </EducationalViz>

      {/* Por que importa */}
      <WhyItMatters>
        <p>
          A atencao e a inovacao central do paper "Attention Is All You Need" (2017).
          Antes dela, modelos usavam RNNs que processavam tokens sequencialmente - o token 1
          precisava ser processado antes do token 2, limitando o paralelismo e esquecendo
          contexto distante.
        </p>
        <p className="mt-2">
          Com atencao, cada token tem acesso direto a todos os outros em O(1), permitindo:
          resolucao de co-referencia ("ele" → "gato"), desambiguacao ("banco" de sentar vs financeiro),
          e relacoes de longa distancia que RNNs nao conseguiam capturar.
        </p>
      </WhyItMatters>
    </div>
  )
}
