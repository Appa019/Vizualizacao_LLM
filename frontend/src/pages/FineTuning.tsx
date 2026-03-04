import { useState } from 'react'
import { Sliders, CheckCircle } from '@phosphor-icons/react'
import PlotlyChart from '../components/viz/PlotlyChart'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import StepByStep from '../components/education/StepByStep'
import Slider from '../components/ui/Slider'

// ─── Fine-tuning ──────────────────────────────────────────────────────────────

export default function FineTuning() {
  const [rank, setRank] = useState(8)

  // Parameter calculations
  const dModel = 4096
  const originalParams = dModel * dModel
  const loraParams = dModel * rank + rank * dModel
  const percentReduction = (
    ((originalParams - loraParams) / originalParams) *
    100
  ).toFixed(2)

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Cabecalho */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Sliders size={12} weight="regular" />
          Modulo 09
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Fine-tuning
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          Fine-tuning e o processo de especializar um modelo base pre-treinado
          para uma tarefa especifica. Em vez de treinar do zero, voce parte de
          um modelo que ja "sabe" linguagem e o ajusta com poucos dados.
        </p>
      </section>

      {/* LoRA Visualization */}
      <section className="glass-card p-6 space-y-5">
        <h3 className="text-sm font-semibold text-gray-800">
          LoRA: Low-Rank Adaptation - Visualizacao
        </h3>

        <Slider
          value={rank}
          onChange={setRank}
          min={1}
          max={64}
          step={1}
          label="Rank (r)"
          showValue
        />

        {/* SVG Matrix Visualization */}
        <div className="flex justify-center">
          <svg
            viewBox="0 0 520 220"
            className="w-full max-w-[520px]"
            style={{ height: 220 }}
          >
            {/* W_0 - Original weight matrix (frozen) */}
            <rect
              x="10"
              y="30"
              width="120"
              height="120"
              rx="6"
              fill="rgba(107,114,128,0.2)"
              stroke="rgba(107,114,128,0.5)"
              strokeWidth="2"
              strokeDasharray="4,4"
            />
            <text
              x="70"
              y="95"
              textAnchor="middle"
              fill="#6b7280"
              fontSize="16"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              W_0
            </text>
            <text
              x="70"
              y="115"
              textAnchor="middle"
              fill="#6b7280"
              fontSize="10"
            >
              {dModel}x{dModel}
            </text>
            <text
              x="70"
              y="170"
              textAnchor="middle"
              fill="#6b7280"
              fontSize="10"
            >
              Congelado
            </text>

            {/* Plus sign */}
            <text
              x="155"
              y="95"
              textAnchor="middle"
              fill="#374151"
              fontSize="24"
              fontWeight="bold"
            >
              +
            </text>

            {/* B matrix (trainable) */}
            <rect
              x="190"
              y="30"
              width={Math.max(24, Math.min(60, rank * 1.5))}
              height="120"
              rx="4"
              fill="rgba(59,130,246,0.25)"
              stroke="rgba(59,130,246,0.6)"
              strokeWidth="2"
            />
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) / 2}
              y="95"
              textAnchor="middle"
              fill="#60a5fa"
              fontSize="14"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              B
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) / 2}
              y="115"
              textAnchor="middle"
              fill="#3b82f6"
              fontSize="9"
            >
              {dModel}x{rank}
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) / 2}
              y="170"
              textAnchor="middle"
              fill="#3b82f6"
              fontSize="10"
            >
              Treinavel
            </text>

            {/* Multiply sign */}
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 18}
              y="95"
              textAnchor="middle"
              fill="#374151"
              fontSize="18"
            >
              x
            </text>

            {/* A matrix (trainable) */}
            <rect
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 36}
              y={90 - Math.max(12, Math.min(30, rank * 0.75))}
              width="120"
              height={Math.max(24, Math.min(60, rank * 1.5))}
              rx="4"
              fill="rgba(34,197,94,0.25)"
              stroke="rgba(34,197,94,0.6)"
              strokeWidth="2"
            />
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 96}
              y="95"
              textAnchor="middle"
              fill="#4ade80"
              fontSize="14"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              A
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 96}
              y="108"
              textAnchor="middle"
              fill="#22c55e"
              fontSize="9"
            >
              {rank}x{dModel}
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 96}
              y="170"
              textAnchor="middle"
              fill="#22c55e"
              fontSize="10"
            >
              Treinavel
            </text>

            {/* = Result */}
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 170}
              y="95"
              textAnchor="middle"
              fill="#374151"
              fontSize="24"
              fontWeight="bold"
            >
              =
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 205}
              y="88"
              textAnchor="middle"
              fill="#f9a8d4"
              fontSize="16"
              fontWeight="bold"
              fontFamily="JetBrains Mono, monospace"
            >
              W
            </text>
            <text
              x={190 + Math.max(24, Math.min(60, rank * 1.5)) + 205}
              y="108"
              textAnchor="middle"
              fill="#ec4899"
              fontSize="10"
            >
              {dModel}x{dModel}
            </text>
          </svg>
        </div>

        {/* Parameter counts */}
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="glass-card p-3">
            <p className="text-xs text-gray-500 mb-1">Params originais</p>
            <p className="text-lg font-bold text-gray-700 font-mono tabular-nums">
              {(originalParams / 1e6).toFixed(1)}M
            </p>
          </div>
          <div className="glass-card p-3">
            <p className="text-xs text-gray-500 mb-1">Params LoRA (r={rank})</p>
            <p className="text-lg font-bold text-blue-400 font-mono tabular-nums">
              {loraParams < 1e6
                ? `${(loraParams / 1e3).toFixed(0)}K`
                : `${(loraParams / 1e6).toFixed(2)}M`}
            </p>
          </div>
          <div className="glass-card p-3">
            <p className="text-xs text-gray-500 mb-1">Reducao</p>
            <p className="text-lg font-bold text-emerald-400 font-mono tabular-nums">
              {percentReduction}%
            </p>
          </div>
        </div>
      </section>

      {/* Formula LoRA */}
      <FormulaBlock
        formula="W = W_0 + BA"
        variables={[
          {
            symbol: 'W',
            color: '#ec4899',
            label: 'Peso final',
            description:
              'Matriz de pesos resultante, combinando o modelo original com a adaptacao LoRA.',
          },
          {
            symbol: 'W_0',
            color: '#9ca3af',
            label: 'Pesos originais (congelados)',
            description:
              'Pesos do modelo pre-treinado. Nao sao alterados durante o fine-tuning com LoRA.',
          },
          {
            symbol: 'B',
            color: '#3b82f6',
            label: 'Matriz B (R^{d x r})',
            description: `Matriz treinavel de dimensao ${dModel} x r. Projeta para o espaco de baixo rank.`,
          },
          {
            symbol: 'A',
            color: '#22c55e',
            label: 'Matriz A (R^{r x d})',
            description: `Matriz treinavel de dimensao r x ${dModel}. Projeta de volta para a dimensao original.`,
          },
        ]}
        size="lg"
      />

      {/* Memory comparison chart */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Comparacao de memoria (Llama 2 7B)
        </h3>
        <div className="glass-card p-4">
          <PlotlyChart
            data={[
              {
                type: 'bar',
                x: ['Full Fine-tuning', 'LoRA', 'QLoRA'],
                y: [60, 16, 6],
                marker: {
                  color: ['#6b7280', '#3b82f6', '#8b5cf6'],
                  opacity: 0.85,
                },
                text: ['~60 GB', '~16 GB', '~6 GB'],
                textposition: 'outside' as const,
                textfont: { color: '#6b7280', size: 12 },
                hovertemplate:
                  '<b>%{x}</b><br>Memoria: %{text}<extra></extra>',
              },
            ]}
            layout={{
              title: {
                text: 'Memoria GPU necessaria (GB)',
                font: { size: 14, color: '#374151' },
              },
              yaxis: {
                title: { text: 'GB VRAM', font: { size: 11 } },
                range: [0, 75],
              },
              showlegend: false,
              annotations: [
                {
                  x: 'Full Fine-tuning',
                  y: 60,
                  text: 'A100 80GB',
                  showarrow: false,
                  yshift: 25,
                  font: { size: 10, color: '#6b7280' },
                },
                {
                  x: 'LoRA',
                  y: 16,
                  text: 'RTX 4090',
                  showarrow: false,
                  yshift: 25,
                  font: { size: 10, color: '#3b82f6' },
                },
                {
                  x: 'QLoRA',
                  y: 6,
                  text: 'RTX 3060',
                  showarrow: false,
                  yshift: 25,
                  font: { size: 10, color: '#8b5cf6' },
                },
              ],
            }}
            height={350}
          />
        </div>
      </section>

      {/* RLHF Formula */}
      <section className="space-y-3">
        <h3 className="text-base font-semibold text-gray-800">
          Objetivo RLHF
        </h3>
        <FormulaBlock
          formula={"\\max_{\\pi} \\mathbb{E}_{x \\sim D}\\left[ R(x, y) - \\beta \\cdot KL(\\pi \\| \\pi_{ref}) \\right]"}
          variables={[
            {
              symbol: 'R',
              color: '#22c55e',
              label: 'Recompensa',
              description:
                'Pontuacao dada pelo modelo de recompensa treinado com preferencias humanas.',
            },
            {
              symbol: '\\pi',
              color: '#3b82f6',
              label: 'Politica',
              description: 'O modelo sendo otimizado (policy).',
            },
            {
              symbol: '\\beta',
              color: '#f59e0b',
              label: 'Coeficiente KL',
              description:
                'Controla quanto o modelo pode divergir do modelo de referencia. Previne "reward hacking".',
            },
          ]}
          size="md"
        />
      </section>

      {/* StepByStep: 3 tecnicas */}
      <StepByStep
        title="Tecnicas de fine-tuning comparadas"
        steps={[
          {
            title: 'Full Fine-tuning',
            description:
              'Atualiza TODOS os parametros do modelo. Maxima flexibilidade e performance, mas requer hardware potente (GPU A100+) e grandes datasets. Risco de catastrophic forgetting.',
            whyItMatters:
              'Use quando performance maxima e critica e voce tem recursos: GPUs de datacenter, milhares de exemplos curados, e budget para experimentacao.',
            content: (
              <div className="flex flex-wrap gap-2">
                {['Melhor performance', 'Adaptacao completa'].map((p) => (
                  <span
                    key={p}
                    className="inline-flex items-center gap-1 text-[11px] text-emerald-700 px-2 py-0.5 bg-emerald-50 border border-emerald-200 rounded"
                  >
                    <CheckCircle size={10} weight="fill" />
                    {p}
                  </span>
                ))}
                {['~60GB VRAM', 'Catastrophic forgetting'].map((c) => (
                  <span
                    key={c}
                    className="text-[11px] text-gray-500 px-2 py-0.5 bg-gray-100 border border-gray-300 rounded"
                  >
                    {c}
                  </span>
                ))}
              </div>
            ),
          },
          {
            title: 'LoRA (Low-Rank Adaptation)',
            description:
              'Congela o modelo original e adiciona matrizes de baixo rank treinaveis. Apenas 0.1-1% dos parametros sao treinados, mas a performance fica proxima do full fine-tuning.',
            whyItMatters:
              'O padrao da industria hoje. Permite treinar em GPUs consumer (RTX 4090), criar multiplos adaptadores para diferentes tarefas, e trocar entre eles sem recarregar o modelo base.',
            content: (
              <div className="flex flex-wrap gap-2">
                {[
                  'Eficiente em memoria',
                  'Multiplos adaptadores',
                  'Sem catastrophic forgetting',
                ].map((p) => (
                  <span
                    key={p}
                    className="inline-flex items-center gap-1 text-[11px] text-emerald-700 px-2 py-0.5 bg-emerald-50 border border-emerald-200 rounded"
                  >
                    <CheckCircle size={10} weight="fill" />
                    {p}
                  </span>
                ))}
                {['Ligeiramente inferior ao full FT'].map((c) => (
                  <span
                    key={c}
                    className="text-[11px] text-gray-500 px-2 py-0.5 bg-gray-100 border border-gray-300 rounded"
                  >
                    {c}
                  </span>
                ))}
              </div>
            ),
          },
          {
            title: 'QLoRA (Quantized LoRA)',
            description:
              'Combina LoRA com quantizacao 4-bit do modelo base. O modelo e carregado em 4 bits (reduzindo VRAM em ~75%), e as matrizes LoRA sao treinadas em fp16. Permite fine-tuning de modelos 70B+ em uma unica GPU.',
            whyItMatters:
              'Democratizou fine-tuning: modelos de 7B cabem em GPUs de 6GB (RTX 3060). Pesquisadores independentes e startups podem criar modelos especializados sem datacenter.',
            content: (
              <div className="flex flex-wrap gap-2">
                {[
                  'Cabe em GPUs consumer',
                  '4-bit quantization',
                  'Quase igual ao LoRA',
                ].map((p) => (
                  <span
                    key={p}
                    className="inline-flex items-center gap-1 text-[11px] text-emerald-700 px-2 py-0.5 bg-emerald-50 border border-emerald-200 rounded"
                  >
                    <CheckCircle size={10} weight="fill" />
                    {p}
                  </span>
                ))}
                {['Treinamento mais lento', 'Overhead de quantizacao'].map(
                  (c) => (
                    <span
                      key={c}
                      className="text-[11px] text-gray-500 px-2 py-0.5 bg-gray-100 border border-gray-300 rounded"
                    >
                      {c}
                    </span>
                  )
                )}
              </div>
            ),
          },
        ]}
      />

      {/* Casos de uso */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Quando fazer fine-tuning?
        </h3>
        <div className="glass-card p-5 space-y-3">
          {[
            {
              quando: 'Dominio especializado',
              exemplo:
                'Medicina, direito, financas - vocabulario e padroes unicos',
            },
            {
              quando: 'Formato de saida',
              exemplo:
                'JSON estruturado, codigo em linguagem especifica, relatorios',
            },
            {
              quando: 'Tom e personalidade',
              exemplo:
                'Chatbot com voz de marca, assistente com persona definida',
            },
            {
              quando: 'Classificacao',
              exemplo:
                'Sentimento, moderacao, categorizacao de tickets',
            },
          ].map((item) => (
            <div
              key={item.quando}
              className="flex gap-3 py-2 border-b border-gray-200 last:border-0"
            >
              <span className="text-xs font-semibold text-pink-400 w-36 flex-shrink-0">
                {item.quando}
              </span>
              <span className="text-xs text-gray-500">{item.exemplo}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Why it matters */}
      <WhyItMatters title="Por que fine-tuning democratiza a IA?">
        <div className="space-y-3">
          <p>
            Antes de LoRA e QLoRA, fine-tuning era privilegio de grandes
            empresas com datacenters de GPUs A100. Hoje, qualquer pessoa com
            uma GPU gamer (RTX 3060+) pode criar um modelo especializado para
            seu dominio.
          </p>
          <p>
            Isso significa que um medico pode criar um assistente de
            triagem, um advogado pode treinar um modelo para analise de
            contratos, e uma startup pode ter seu proprio modelo de
            atendimento - tudo com custos acessiveis e dados proprietarios
            que nunca saem da organizacao.
          </p>
          <p>
            A combinacao de modelos base open-source (Llama, Mistral) com
            tecnicas eficientes de adaptacao (QLoRA) esta criando um ecossistema
            onde especializacao supera escala bruta.
          </p>
        </div>
      </WhyItMatters>
    </div>
  )
}
