import { useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Brain, BookOpenText, Scales, SquaresFour } from '@phosphor-icons/react'
import PlotlyChart from '../components/viz/PlotlyChart'
import PredictNextWord from '../components/viz/PredictNextWord'
import StepByStep from '../components/education/StepByStep'
import WhyItMatters from '../components/education/WhyItMatters'
import ConteudoAdaptavel from '../components/education/ConteudoAdaptavel'
import type { LayoutContext } from '../components/layout/Layout'

// ─── O que sao LLMs? ──────────────────────────────────────────────────────────

const timelineItems = [
  {
    ano: 1986,
    nome: 'RNN',
    cor: '#6b7280',
    descricao:
      'Redes Neurais Recorrentes processam sequencias token por token, mantendo um "estado oculto" que carrega informacao dos tokens anteriores. Limitacao: dificuldade com dependencias de longo alcance.',
    impacto: 'Primeira arquitetura para sequencias',
  },
  {
    ano: 1997,
    nome: 'LSTM',
    cor: '#8b5cf6',
    descricao:
      'Long Short-Term Memory adicionou "portoes" (gates) que controlam o fluxo de informacao, permitindo memorizar dependencias mais longas. Ainda sequencial, mas muito mais robusto que RNNs.',
    impacto: 'Permitiu traducao automatica neural',
  },
  {
    ano: 2017,
    nome: 'Transformer',
    cor: '#ef4444',
    descricao:
      '"Attention Is All You Need" (Vaswani et al.) introduziu self-attention, eliminando recorrencia. Cada token pode olhar para todos os outros em paralelo -- revolucionando NLP.',
    impacto: 'Revolucionou todo o campo de NLP',
  },
  {
    ano: 2019,
    nome: 'GPT-2',
    cor: '#22c55e',
    descricao:
      'OpenAI treinou um Transformer decoder-only com 1.5B parametros em 40GB de texto da web. Tao bom em gerar texto que inicialmente nao foi liberado por medo de uso indevido.',
    impacto: 'Mostrou que escala importa',
  },
  {
    ano: 2020,
    nome: 'GPT-3',
    cor: '#3b82f6',
    descricao:
      'Com 175B parametros, demonstrou capacidades emergentes: few-shot learning, raciocinio basico, geracao de codigo. Mostrou que escala gera capacidades nao previstas.',
    impacto: 'Capacidades emergentes surpreenderam pesquisadores',
  },
  {
    ano: 2022,
    nome: 'ChatGPT',
    cor: '#f59e0b',
    descricao:
      'GPT-3.5 ajustado com RLHF para seguir instrucoes e dialogar. Atingiu 100M de usuarios em 2 meses -- o crescimento mais rapido da historia de produtos de consumo.',
    impacto: '100M usuarios em 2 meses',
  },
  {
    ano: 2023,
    nome: 'GPT-4',
    cor: '#ec4899',
    descricao:
      'Modelo multimodal (texto + imagem) com ~1.8T parametros (estimado). Passou no exame da OAB, em provas de medicina e em competicoes de programacao. Marco da IA generalista.',
    impacto: 'IA generalista se tornou realidade',
  },
]

const modelParams = {
  nomes: ['GPT-2', 'GPT-3', 'Llama 2 7B', 'Llama 2 70B', 'GPT-4'],
  valores: [117e6, 175e9, 7e9, 70e9, 1.8e12],
  cores: ['#22c55e', '#3b82f6', '#f59e0b', '#f59e0b', '#ec4899'],
}

export default function WhatAreLLMs() {
  const { nivelConhecimento } = useOutletContext<LayoutContext>()
  const [selectedTimeline, setSelectedTimeline] = useState<number | null>(null)

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Introducao */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Brain size={12} weight="duotone" />
          Modulo 02
        </div>

        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          O que sao LLMs?
        </h2>

        <ConteudoAdaptavel
          avancado={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              Grandes Modelos de Linguagem (LLMs -- <em>Large Language Models</em>)
              sao sistemas de inteligencia artificial treinados em enormes volumes
              de texto. Eles aprendem padroes estatisticos da linguagem humana e
              conseguem gerar, resumir, traduzir e raciocinar sobre texto com
              precisao notavel.
            </p>
          }
          iniciante={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              Imagine que voce pudesse ler todos os livros, sites e conversas do mundo. Depois de ler tudo isso, voce comeca a perceber padroes - quais palavras aparecem juntas, como frases sao construidas, como ideias se conectam. LLMs fazem exatamente isso, mas com matematica: eles leem bilhoes de textos e aprendem padroes estatisticos da linguagem.
            </p>
          }
        />
      </section>

      {/* Cards de definicao */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          {
            icone: Scales,
            titulo: 'Escala massiva',
            descricao:
              'Bilhoes de parametros treinados em trilhoes de tokens -- texto da internet, livros, codigo e muito mais.',
            cor: 'text-blue-600',
          },
          {
            icone: SquaresFour,
            titulo: 'Arquitetura Transformer',
            descricao:
              'Todos os LLMs modernos sao baseados no Transformer, proposto no artigo "Attention Is All You Need" (2017).',
            cor: 'text-violet-600',
          },
          {
            icone: BookOpenText,
            titulo: 'Aprendizado por previsao',
            descricao:
              'O objetivo de treinamento mais comum e prever o proximo token -- simples, mas incrivelmente poderoso.',
            cor: 'text-emerald-600',
          },
        ].map((item) => {
          const Icone = item.icone
          return (
            <div key={item.titulo} className="glass-card p-5">
              <Icone size={20} weight="duotone" className={`${item.cor} mb-3`} />
              <h3 className="text-sm font-semibold text-gray-800 mb-2">
                {item.titulo}
              </h3>
              <p className="text-xs text-gray-500 leading-relaxed">
                {item.descricao}
              </p>
            </div>
          )
        })}
      </section>

      <ConteudoAdaptavel
        avancado={null}
        iniciante={
          <section className="glass-card p-5 bg-green-50 border-green-200">
            <h4 className="text-sm font-semibold text-green-700 mb-2">O que e um parametro?</h4>
            <p className="text-xs text-green-600 leading-relaxed">
              Imagine um mixer de som com bilhoes de botoes. Cada botao controla um aspecto minusculo de como o modelo entende linguagem - um botao afeta como ele entende "gato", outro como liga "gato" a "animal". "Treinar" e ajustar todos esses botoes ate o som ficar perfeito.
            </p>
          </section>
        }
      />

      {/* Timeline horizontal */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Evolucao dos modelos de linguagem
        </h3>
        <div className="glass-card p-6">
          {/* Timeline bar */}
          <div className="relative">
            <div className="flex items-center justify-between overflow-x-auto pb-2">
              {timelineItems.map((item, i) => (
                <button
                  key={item.ano}
                  onClick={() =>
                    setSelectedTimeline(selectedTimeline === i ? null : i)
                  }
                  className={`flex flex-col items-center gap-2 px-3 py-2 rounded-sm transition-all min-w-[80px] ${
                    selectedTimeline === i
                      ? 'bg-gray-100 ring-1 ring-gray-300'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <div
                    className="w-3 h-3 rounded-full ring-2 ring-offset-2 ring-offset-white transition-transform"
                    style={{
                      backgroundColor: item.cor,
                      outlineColor: item.cor,
                      transform:
                        selectedTimeline === i ? 'scale(1.3)' : 'scale(1)',
                    }}
                  />
                  <span
                    className="text-xs font-semibold"
                    style={{
                      color:
                        selectedTimeline === i ? item.cor : '#6b7280',
                    }}
                  >
                    {item.nome}
                  </span>
                  <span className="text-[10px] text-gray-400">{item.ano}</span>
                </button>
              ))}
            </div>

            {/* Connecting line */}
            <div className="absolute top-[18px] left-[40px] right-[40px] h-px bg-gray-300 -z-10" />
          </div>

          {/* Description panel */}
          {selectedTimeline !== null && (
            <div
              className="mt-4 p-4 rounded-sm bg-gray-100 border border-gray-200 animate-slide-up border-l-[3px]"
              style={{ borderLeftColor: timelineItems[selectedTimeline].cor }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{
                    backgroundColor: timelineItems[selectedTimeline].cor,
                  }}
                />
                <span className="text-sm font-semibold text-gray-800">
                  {timelineItems[selectedTimeline].nome} (
                  {timelineItems[selectedTimeline].ano})
                </span>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed">
                {timelineItems[selectedTimeline].descricao}
              </p>
              <p
                className="mt-2 text-[11px] font-semibold tracking-wide"
                style={{ color: timelineItems[selectedTimeline].cor }}
              >
                Impacto: {timelineItems[selectedTimeline].impacto}
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Grafico de parametros */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Escala de parametros (escala logaritmica)
        </h3>
        <div className="glass-card p-4">
          <PlotlyChart
            data={[
              {
                type: 'bar',
                x: modelParams.nomes,
                y: modelParams.valores,
                marker: {
                  color: modelParams.cores,
                  opacity: 0.85,
                },
                text: ['117M', '175B', '7B', '70B', '~1.8T'],
                textposition: 'outside' as const,
                textfont: { color: '#6b7280', size: 11 },
                hovertemplate:
                  '<b>%{x}</b><br>Parametros: %{text}<extra></extra>',
              },
            ]}
            layout={{
              title: {
                text: 'Numero de parametros por modelo',
                font: { size: 14, color: '#374151' },
              },
              yaxis: {
                type: 'log',
                title: {
                  text: 'Parametros (escala log)',
                  font: { size: 11 },
                },
                gridcolor: 'rgba(209,213,219,0.5)',
              },
              xaxis: {
                title: { text: '' },
              },
              showlegend: false,
            }}
            height={350}
          />
        </div>
      </section>

      {/* Step by Step: como previsao vira inteligencia */}
      <ConteudoAdaptavel
        avancado={null}
        iniciante={
          <section className="glass-card p-5 bg-green-50 border-green-200 mb-4">
            <h4 className="text-sm font-semibold text-green-700 mb-2">Uma analogia sobre emergencia</h4>
            <p className="text-xs text-green-600 leading-relaxed">
              Uma formiga sozinha nao faz nada impressionante. Mas bilhoes de formigas juntas constroem cidades inteiras. LLMs funcionam assim - bilhoes de parametros simples, juntos, criam comportamento inteligente.
            </p>
          </section>
        }
      />
      <StepByStep
        title="Como previsao de proximo token vira inteligencia"
        steps={[
          {
            title: 'Passo 1: Previsao estatistica',
            description:
              'O modelo aprende padroes estatisticos da linguagem ao prever o proximo token em bilhoes de frases. Palavras que frequentemente aparecem juntas formam associacoes fortes nos pesos da rede.',
            whyItMatters:
              'Prever "O gato sentou no ___" parece simples, mas requer entender gramatica, semantica e contexto.',
            content: (
              <div className="flex items-center gap-2 flex-wrap font-mono text-xs">
                {['O', 'gato', 'sentou', 'no'].map((t) => (
                  <span
                    key={t}
                    className="px-2 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded"
                  >
                    {t}
                  </span>
                ))}
                <span className="px-2 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-sm">
                  tapete
                </span>
                <span className="text-gray-400 ml-2">P = 0.73</span>
              </div>
            ),
          },
          {
            title: 'Passo 2: Emergencia',
            description:
              'Com escala (mais parametros + mais dados), surgem capacidades que nao foram explicitamente treinadas: raciocinio logico, traducao, resolucao de problemas matematicos, geracao de codigo.',
            whyItMatters:
              'GPT-3 aprendeu a fazer aritmetica basica sem nunca ter sido treinado especificamente para isso -- so prevendo tokens.',
            content: (
              <div className="grid grid-cols-2 gap-2 text-xs">
                {[
                  { cap: 'Raciocinio', escala: '> 10B params' },
                  { cap: 'Codigo', escala: '> 50B params' },
                  { cap: 'Matematica', escala: '> 100B params' },
                  { cap: 'Chain-of-thought', escala: '> 500B params' },
                ].map((c) => (
                  <div
                    key={c.cap}
                    className="px-3 py-2 bg-violet-50 border border-violet-200 rounded-sm"
                  >
                    <span className="text-violet-700 font-medium">
                      {c.cap}
                    </span>
                    <span className="text-gray-500 ml-2">{c.escala}</span>
                  </div>
                ))}
              </div>
            ),
          },
          {
            title: 'Passo 3: Instrucao (RLHF)',
            description:
              'Reinforcement Learning from Human Feedback alinha o modelo com intencoes humanas. Avaliadores humanos classificam respostas, e o modelo aprende a gerar as preferidas.',
            whyItMatters:
              'Sem RLHF, o modelo so completa texto. Com RLHF, ele segue instrucoes, recusa pedidos perigosos e dialoga naturalmente.',
            content: (
              <div className="flex flex-col gap-2 text-xs">
                <div className="flex items-center gap-3">
                  <span className="w-20 text-gray-500">Pre-treino:</span>
                  <span className="text-gray-700">
                    "O gato..." -&gt; "sentou no tapete"
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-20 text-gray-500">SFT:</span>
                  <span className="text-gray-700">
                    "Explique X" -&gt; resposta estruturada
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-20 text-gray-500">RLHF:</span>
                  <span className="text-emerald-700">
                    "Explique X" -&gt; resposta preferida por humanos
                  </span>
                </div>
              </div>
            ),
          },
        ]}
      />

      {/* Tabela de LLMs */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          LLMs que voce provavelmente conhece
        </h3>
        <div className="glass-card divide-y divide-gray-200">
          {[
            {
              nome: 'GPT-4',
              empresa: 'OpenAI',
              parametros: '~1.8T (estimado)',
              uso: 'ChatGPT, API',
            },
            {
              nome: 'Llama 3',
              empresa: 'Meta',
              parametros: '8B - 70B',
              uso: 'Open-source',
            },
            {
              nome: 'Gemini',
              empresa: 'Google DeepMind',
              parametros: 'Ultra, Pro, Flash',
              uso: 'Google Products',
            },
            {
              nome: 'LLaMA 3',
              empresa: 'Meta AI',
              parametros: '8B, 70B, 405B',
              uso: 'Open source, API',
            },
          ].map((llm) => (
            <div
              key={llm.nome}
              className="flex items-center justify-between px-5 py-3.5"
            >
              <div>
                <p className="text-sm font-medium text-gray-800">{llm.nome}</p>
                <p className="text-xs text-gray-500">{llm.empresa}</p>
              </div>
              <div className="text-right hidden sm:block">
                <p className="text-xs text-gray-600 font-mono">
                  {llm.parametros}
                </p>
                <p className="text-xs text-gray-400">{llm.uso}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Why it matters */}
      <WhyItMatters title="Por que os LLMs mudaram a IA para sempre?">
        <div className="space-y-3">
          <p>
            LLMs representam uma mudanca de paradigma: ao inves de programar
            regras explicitas para cada tarefa, um unico modelo treinado em
            texto generico consegue realizar centenas de tarefas diferentes
            apenas recebendo instrucoes em linguagem natural.
          </p>
          <p>
            Isso democratizou o acesso a IA. Antes, era necessario treinar um
            modelo especifico para cada problema (classificacao de sentimento,
            traducao, resumo). Agora, um unico LLM faz tudo isso e mais --
            bastando descrever o que voce quer em portugues.
          </p>
        </div>
      </WhyItMatters>

      {/* Jogo de predicao */}
      <PredictNextWord />

      {/* Proximo modulo */}
      <div className="glass-card p-5 flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-500 mb-1">Proximo modulo</p>
          <p className="text-sm font-semibold text-gray-800">
            Tokenizacao -- como o texto vira numeros
          </p>
        </div>
        <a href="/tokenization" className="btn-primary text-sm">
          Continuar
        </a>
      </div>
    </div>
  )
}
