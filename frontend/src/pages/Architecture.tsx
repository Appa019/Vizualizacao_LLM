import { useState } from 'react'
import { useOutletContext } from 'react-router-dom'
import { Cpu } from '@phosphor-icons/react'
import NetworkGraph from '../components/viz/NetworkGraph'
import TokenFlow3D from '../components/viz/TokenFlow3D'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import ConteudoAdaptavel from '../components/education/ConteudoAdaptavel'
import type { LayoutContext } from '../components/layout/Layout'

// ─── Arquitetura Transformer ──────────────────────────────────────────────────

interface LayerInfo {
  name: string
  type: 'input' | 'embedding' | 'attention' | 'ffn' | 'norm' | 'output'
  description: string
  color: string
}

const layerFormulas: Record<string, { formula: string; explanation: string }> = {
  attention: {
    formula: 'MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O',
    explanation:
      'Cada cabeca de atencao processa Q, K e V independentemente. Os resultados sao concatenados e projetados por W^O para a dimensao original.',
  },
  ffn: {
    formula: 'FFN(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
    explanation:
      'A rede feed-forward expande a dimensao para 4x d_model (com ReLU) e comprime de volta. E onde o modelo "memoriza" conhecimento factual.',
  },
  norm: {
    formula:
      'LayerNorm(x) = \\gamma \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta',
    explanation:
      'Normaliza as ativacoes para media 0 e variancia 1, depois aplica escala (gamma) e deslocamento (beta) aprendiveis. Estabiliza o treinamento.',
  },
  embedding: {
    formula: 'E(x) = TokenEmb(x) + PosEnc(pos)',
    explanation:
      'A embedding combina a representacao do token com informacao posicional, permitindo ao modelo saber a ordem das palavras.',
  },
  input: {
    formula: 'x \\in \\{0, 1, ..., V-1\\}',
    explanation:
      'Os tokens de entrada sao indices inteiros do vocabulario, mapeados para vetores pela camada de embedding.',
  },
  output: {
    formula: 'P(w) = \\text{softmax}(xW_{out} + b)',
    explanation:
      'A camada de saida projeta os hidden states para o tamanho do vocabulario e aplica softmax para obter probabilidades.',
  },
}

const tensorShapes = [
  { etapa: 'Input', forma: '[seq_len]', descricao: 'Indices de tokens' },
  {
    etapa: 'Apos Embedding',
    forma: '[seq_len, d_model]',
    descricao: 'Vetores densos',
  },
  {
    etapa: 'Apos Attention',
    forma: '[seq_len, d_model]',
    descricao: 'Contexto agregado',
  },
  {
    etapa: 'FFN (expansao)',
    forma: '[seq_len, 4*d_model]',
    descricao: 'Dimensao expandida',
  },
  {
    etapa: 'FFN (compressao)',
    forma: '[seq_len, d_model]',
    descricao: 'De volta a d_model',
  },
  {
    etapa: 'Output',
    forma: '[seq_len, vocab_size]',
    descricao: 'Logits sobre vocabulario',
  },
]

const hyperparameters = [
  {
    modelo: 'GPT-2 Small',
    d_model: 768,
    n_heads: 12,
    d_ff: 3072,
    n_layers: 12,
    vocab: '50.257',
    params: '117M',
  },
  {
    modelo: 'GPT-3',
    d_model: 12288,
    n_heads: 96,
    d_ff: 49152,
    n_layers: 96,
    vocab: '50.257',
    params: '175B',
  },
  {
    modelo: 'Llama 2 7B',
    d_model: 4096,
    n_heads: 32,
    d_ff: 11008,
    n_layers: 32,
    vocab: '32.000',
    params: '7B',
  },
  {
    modelo: 'Llama 2 70B',
    d_model: 8192,
    n_heads: 64,
    d_ff: 28672,
    n_layers: 80,
    vocab: '32.000',
    params: '70B',
  },
]

export default function Architecture() {
  const { nivelConhecimento } = useOutletContext<LayoutContext>()
  const [selectedLayer, setSelectedLayer] = useState<LayerInfo | null>(null)

  const handleLayerClick = (layer: LayerInfo) => {
    setSelectedLayer(layer)
  }

  const selectedFormula = selectedLayer
    ? layerFormulas[selectedLayer.type]
    : null

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Cabecalho */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Cpu size={12} weight="duotone" />
          Modulo 05
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Arquitetura Transformer
        </h2>
        <ConteudoAdaptavel
          avancado={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              O Transformer foi introduzido em 2017 no artigo "Attention Is All You
              Need" por Vaswani et al. Ele substituiu recorrencia (RNNs, LSTMs) por
              atencao, permitindo paralelizacao massiva e captura de dependencias de
              longo alcance.
            </p>
          }
          iniciante={
            <p className="text-gray-600 leading-relaxed max-w-2xl">
              O Transformer e como uma fabrica com varias estacoes de trabalho empilhadas. Cada estacao (camada) faz algo diferente com o texto: uma entende gramatica, outra captura significado, outra analisa contexto. O texto passa por todas as estacoes e sai 'entendido'.
            </p>
          }
        />
      </section>

      <ConteudoAdaptavel
        avancado={null}
        iniciante={
          <div className="space-y-4">
            <section className="glass-card p-5 bg-green-50 border-green-200">
              <h4 className="text-sm font-semibold text-green-700 mb-2">O que e uma camada?</h4>
              <p className="text-xs text-green-600 leading-relaxed">
                Pense em filtros do Instagram empilhados. Cada filtro transforma a imagem de um jeito. No Transformer, cada camada transforma a compreensao do texto - uma ve gramatica, outra ve significado, outra ve contexto.
              </p>
            </section>
            <section className="glass-card p-5 bg-blue-50 border-blue-200">
              <h4 className="text-sm font-semibold text-blue-700 mb-2">Por que Transformers sao mais rapidos?</h4>
              <p className="text-xs text-blue-600 leading-relaxed">
                RNNs leem como humanos - uma palavra por vez, na ordem. Transformers leem como quem faz prova - olham tudo de uma vez e voltam no que importa. Por isso sao muito mais rapidos e podem processar textos enormes.
              </p>
            </section>
          </div>
        }
      />

      {/* Visualizacao 3D do Transformer */}
      <EducationalViz
        title="Arquitetura Transformer (Decoder-only)"
        icon={<Cpu size={18} weight="duotone" />}
        caption="Clique em uma camada para ver detalhes. Arraste para rotacionar a visualizacao 3D."
        formula={{
          formula:
            'Transformer(x) = Decoder_N(\\cdots Decoder_1(Emb(x) + PE))',
          variables: [
            {
              symbol: 'x',
              color: '#3b82f6',
              label: 'Input',
              description: 'Sequencia de tokens de entrada',
            },
            {
              symbol: 'N',
              color: '#f59e0b',
              label: 'Camadas',
              description: 'Numero de blocos decoder empilhados',
            },
          ],
        }}
        whyItMatters={
          <p>
            A arquitetura Transformer e a base de todos os LLMs modernos. Sua
            capacidade de processar tokens em paralelo (ao inves de
            sequencialmente como RNNs) permitiu escalar modelos para bilhoes de
            parametros, resultando em capacidades emergentes como raciocinio e
            geração de codigo.
          </p>
        }
      >
        <NetworkGraph onLayerClick={handleLayerClick} height={500} />
      </EducationalViz>

      {/* Painel de detalhes da camada selecionada */}
      {selectedLayer && selectedFormula && (
        <section className="glass-card p-6 space-y-4 animate-slide-up">
          <div className="flex items-center gap-3">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: selectedLayer.color }}
            />
            <h3 className="text-base font-semibold text-gray-800">
              {selectedLayer.name.replace('\n', ' ')}
            </h3>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
              {selectedLayer.type}
            </span>
          </div>

          <p className="text-sm text-gray-600 leading-relaxed">
            {selectedLayer.description}
          </p>

          <FormulaBlock
            formula={selectedFormula.formula}
            size="md"
          />

          <p className="text-xs text-gray-500 leading-relaxed">
            {selectedFormula.explanation}
          </p>
        </section>
      )}

      {/* Fluxo de tokens pelas camadas */}
      <EducationalViz
        title="Fluxo de Tokens pelas Camadas"
        icon={<Cpu size={18} weight="duotone" />}
        caption="Observe como os tokens fluem por cada camada do Transformer, mudando de representacao em cada estagio."
      >
        <TokenFlow3D />
      </EducationalViz>

      {/* Tensor shapes */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Formas dos tensores ao longo do modelo
        </h3>
        <div className="glass-card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left px-5 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Etapa
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Forma
                </th>
                <th className="text-left px-5 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider hidden sm:table-cell">
                  Descricao
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {tensorShapes.map((row) => (
                <tr key={row.etapa} className="hover:bg-gray-100">
                  <td className="px-5 py-2.5 text-gray-700 text-xs">
                    {row.etapa}
                  </td>
                  <td className="px-5 py-2.5 font-mono text-orange-400 text-xs">
                    {row.forma}
                  </td>
                  <td className="px-5 py-2.5 text-gray-500 text-xs hidden sm:table-cell">
                    {row.descricao}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Hiperparametros de modelos populares */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">
          Hiperparametros de modelos populares
        </h3>
        <div className="glass-card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Modelo
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  d_model
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Heads
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  d_ff
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Layers
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Vocab
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-600 uppercase tracking-wider">
                  Params
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {hyperparameters.map((m) => (
                <tr key={m.modelo} className="hover:bg-gray-100">
                  <td className="px-4 py-2.5 text-gray-800 text-xs font-medium">
                    {m.modelo}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-orange-400 text-xs">
                    {m.d_model.toLocaleString()}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-gray-700 text-xs">
                    {m.n_heads}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-gray-700 text-xs">
                    {m.d_ff.toLocaleString()}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-gray-700 text-xs">
                    {m.n_layers}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-gray-700 text-xs">
                    {m.vocab}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-emerald-400 text-xs font-semibold">
                    {m.params}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Why it matters */}
      <WhyItMatters title="Por que a arquitetura Transformer revolucionou a IA?">
        <div className="space-y-3">
          <p>
            Antes do Transformer, modelos de linguagem usavam RNNs e LSTMs que
            processavam tokens sequencialmente -- cada token dependia do
            anterior. Isso criava dois problemas graves: treinamento lento
            (impossivel paralelizar) e perda de informacao em sequencias longas.
          </p>
          <p>
            O mecanismo de Self-Attention resolve ambos: cada token pode
            "olhar" diretamente para qualquer outro token na sequencia,
            independente da distancia. E como todo o processamento e feito em
            paralelo, GPUs modernas conseguem treinar modelos com bilhoes de
            parametros em semanas ao inves de anos.
          </p>
          <p>
            Essa combinacao de paralelismo + atencao global e o que permitiu a
            era dos LLMs: GPT-3 (175B), Llama (70B), e modelos ainda maiores
            so existem gracas a essa arquitetura.
          </p>
        </div>
      </WhyItMatters>
    </div>
  )
}
