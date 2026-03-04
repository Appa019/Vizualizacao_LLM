import { useState, useEffect } from 'react'
import { Stack } from '@phosphor-icons/react'
import { useEmbeddings, usePositionalEncoding, useEmbeddingSpace, useTokenize } from '../api/hooks'
import EmbeddingSpace from '../components/viz/EmbeddingSpace'
import Heatmap3D from '../components/viz/Heatmap3D'
import StepByStep from '../components/education/StepByStep'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import Toggle from '../components/ui/Toggle'
import ApiLoadingState from '../components/education/ApiLoadingState'

// ─── Dados fallback ─────────────────────────────────────────────────────────

const FALLBACK_POINTS = [
  { word: 'rei', position: [1.2, 1.5, 0.3] as [number, number, number], category: 'substantivo' },
  { word: 'rainha', position: [1.0, 1.3, 0.1] as [number, number, number], category: 'substantivo' },
  { word: 'gato', position: [-1.0, -0.5, 0.8] as [number, number, number], category: 'animal' },
  { word: 'cachorro', position: [-0.8, -0.7, 0.6] as [number, number, number], category: 'animal' },
  { word: 'paris', position: [0.5, -1.2, -1.0] as [number, number, number], category: 'substantivo' },
]

const PALAVRAS_TESTE = ['gato', 'cachorro', 'rei', 'rainha', 'homem', 'mulher', 'paris', 'londres', 'feliz', 'triste']

// ─── Componente principal ───────────────────────────────────────────────────

export default function Embeddings() {
  const [inputText, setInputText] = useState('O gato dormiu no tapete quente')
  const [metodo, setMetodo] = useState<'pca' | 'tsne'>('pca')

  const tokenizer = useTokenize()
  const embeddings = useEmbeddings()
  const posEncoding = usePositionalEncoding()
  const embSpace = useEmbeddingSpace()

  // Tokenizar e buscar embeddings
  useEffect(() => {
    tokenizer.execute({ texto: inputText })
  }, [inputText])

  useEffect(() => {
    if (tokenizer.data) {
      embeddings.execute({ tokens: tokenizer.data.tokens, d_model: 64 })
    }
  }, [tokenizer.data])

  // Buscar positional encoding
  useEffect(() => {
    posEncoding.execute({ seq_length: 20, d_model: 64 })
  }, [])

  // Buscar embedding space com palavras de teste
  useEffect(() => {
    embSpace.execute({ tokens: PALAVRAS_TESTE, metodo, d_model: 64 })
  }, [metodo])

  // StepByStep: Token Lookup -> PE -> Soma
  const steps = [
    {
      title: '1. Token Embedding Lookup',
      description: 'Cada token e convertido em um vetor denso de dimensao d_model via uma tabela de lookup treinavel. Esse vetor captura o significado semantico da palavra.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-gray-50 rounded-sm p-3">
          E_token = EmbeddingTable[token_id]  →  vetor de {embeddings.data?.d_model ?? 64} dimensoes
        </div>
      ),
    },
    {
      title: '2. Positional Encoding',
      description: 'Como o Transformer processa todos os tokens em paralelo (sem recorrencia), precisamos adicionar informacao de posicao. Usamos funcoes seno e cosseno de frequencias diferentes.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-gray-50 rounded-sm p-3">
          PE(pos, 2i) = sin(pos / 10000^(2i/d_model)){'\n'}
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        </div>
      ),
      whyItMatters: 'Sem positional encoding, "O gato comeu o rato" e "O rato comeu o gato" seriam identicos para o modelo!',
    },
    {
      title: '3. Soma: Embedding Final',
      description: 'O embedding final e a soma do token embedding com o positional encoding. Isso combina o significado da palavra com sua posicao na sequencia.',
      content: (
        <div className="text-xs font-mono text-gray-500 bg-gray-50 rounded-sm p-3">
          X = E_token + PE  →  entrada para o bloco Transformer
        </div>
      ),
    },
  ]

  // Heatmap data do positional encoding
  const peMatrix = posEncoding.data?.encoding?.slice(0, 15).map((r) => r.slice(0, 20)) ?? []
  const peXLabels = Array.from({ length: 20 }, (_, i) => `d${i}`)
  const peYLabels = Array.from({ length: 15 }, (_, i) => `pos ${i}`)

  // Points do embedding space
  const spacePoints = embSpace.data?.pontos ?? FALLBACK_POINTS

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Header */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Stack size={12} weight="duotone" />
          Modulo 04
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Embeddings
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          Um embedding e uma representacao numerica de um token em um espaco vetorial de
          alta dimensao. Tokens com significados semelhantes ficam proximos nesse espaco -
          e assim que o modelo "entende" semantica.
        </p>
      </section>

      {/* Input */}
      <section className="glass-card p-5">
        <label className="text-sm font-medium text-gray-700 mb-1 block">
          Texto de entrada
        </label>
        <p className="text-[11px] text-gray-400 mb-2">Exemplo fixo para demonstracao</p>
        <input
          type="text"
          value={inputText}
          readOnly
          className="w-full bg-gray-50 border border-gray-200 rounded-sm px-4 py-2.5 text-sm text-gray-900 cursor-default"
        />
      </section>

      {/* StepByStep */}
      <StepByStep steps={steps} title="Como Embeddings Sao Construidos" />

      {/* Formula KaTeX */}
      <FormulaBlock
        formula={"PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)"}
        variables={[
          { symbol: 'pos', color: '#a855f7', label: 'Posicao', description: 'Posicao do token na sequencia (0, 1, 2, ...)' },
          { symbol: 'i', color: '#3b82f6', label: 'Dimensao', description: 'Indice da dimensao do embedding' },
          { symbol: 'd_{model}', color: '#f59e0b', label: 'Dimensao do modelo', description: 'Tamanho total do vetor de embedding' },
        ]}
      />

      {/* Embedding Space 3D */}
      <EducationalViz
        title="Espaco de Embeddings 3D"
        icon={<Stack size={18} weight="duotone" />}
        caption="Palavras semanticamente similares ficam proximas no espaco vetorial. Arraste para rotacionar."
        whyItMatters={
          <span>
            A propriedade mais poderosa dos embeddings e que relacoes semanticas viram operacoes
            aritmeticas: rei - homem + mulher ≈ rainha. Isso funciona porque a "direcao" de genero
            e consistente no espaco vetorial.
          </span>
        }
      >
        <div className="space-y-3">
          <div className="flex items-center justify-end gap-3">
            <Toggle
              enabled={metodo === 'tsne'}
              onChange={(v) => setMetodo(v ? 'tsne' : 'pca')}
              labelLeft="PCA"
              labelRight="t-SNE"
              size="sm"
            />
          </div>
          <ApiLoadingState
            loading={embSpace.loading}
            error={embSpace.error}
            fallback={<EmbeddingSpace points={FALLBACK_POINTS} height={450} />}
          >
            <EmbeddingSpace points={spacePoints} height={450} />
          </ApiLoadingState>

          {/* Legenda de categorias */}
          <div className="flex flex-wrap gap-4 pt-2">
            {[
              { cat: 'animal', color: 'bg-green-500', label: 'Animal' },
              { cat: 'cor', color: 'bg-violet-500', label: 'Cor' },
              { cat: 'verbo', color: 'bg-blue-500', label: 'Verbo' },
              { cat: 'substantivo', color: 'bg-orange-500', label: 'Substantivo' },
            ].map((item) => (
              <div key={item.cat} className="flex items-center gap-1.5">
                <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                <span className="text-[11px] text-gray-600">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      </EducationalViz>

      {/* Positional Encoding Heatmap */}
      <EducationalViz
        title="Positional Encoding - Padrao Sinusoidal"
        icon={<Stack size={18} weight="duotone" />}
        caption="Cada posicao tem um padrao unico de senos e cossenos. Frequencias diferentes capturam relacoes de diferentes escalas."
      >
        <ApiLoadingState
          loading={posEncoding.loading}
          error={posEncoding.error}
        >
          {peMatrix.length > 0 && (
            <Heatmap3D
              matrix={peMatrix}
              xLabels={peXLabels}
              yLabels={peYLabels}
              title="Positional Encoding (posicao x dimensao)"
              colorscale="RdBu"
              height={350}
            />
          )}
        </ApiLoadingState>
      </EducationalViz>

      {/* Analogias vetoriais */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">A aritmetica dos embeddings</h3>
        <div className="glass-card p-5">
          <div className="flex flex-wrap items-center gap-3 font-mono text-sm">
            <span className="px-3 py-1.5 rounded bg-blue-50 border border-blue-200 text-blue-700">rei</span>
            <span className="text-gray-400">-</span>
            <span className="px-3 py-1.5 rounded bg-blue-50 border border-blue-200 text-blue-700">homem</span>
            <span className="text-gray-400">+</span>
            <span className="px-3 py-1.5 rounded bg-pink-50 border border-pink-200 text-pink-700">mulher</span>
            <span className="text-gray-400">=</span>
            <span className="px-3 py-1.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 font-semibold">rainha</span>
          </div>
          <p className="text-xs text-gray-500 mt-4 leading-relaxed">
            Relacoes semanticas se tornam operacoes aritmeticas no espaco vetorial.
            A "direcao" de genero e consistente - funciona para qualquer par.
          </p>
        </div>
      </section>

      {/* Dimensoes em LLMs */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">Dimensoes em LLMs reais</h3>
        <div className="glass-card divide-y divide-gray-200">
          {[
            { modelo: 'GPT-2 (small)', params: '117M', dim: 768 },
            { modelo: 'GPT-3', params: '175B', dim: 12288 },
            { modelo: 'Llama 2 7B', params: '7B', dim: 4096 },
            { modelo: 'Llama 2 70B', params: '70B', dim: 8192 },
          ].map((r) => (
            <div key={r.modelo} className="flex items-center justify-between px-5 py-3">
              <div>
                <p className="text-sm text-gray-800 font-medium">{r.modelo}</p>
                <p className="text-xs text-gray-500">{r.params} parametros</p>
              </div>
              <div className="text-right">
                <p className="text-sm font-mono text-violet-600 tabular-nums">{r.dim.toLocaleString('pt-BR')}</p>
                <p className="text-xs text-gray-400">dimensoes</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <WhyItMatters>
        Embeddings sao a base de tudo em LLMs - sao a forma como texto vira matematica.
        Cada token e representado por um vetor denso onde cada dimensao captura uma faceta
        do significado. Sem bons embeddings, nenhuma camada subsequente consegue entender
        o texto.
      </WhyItMatters>
    </div>
  )
}
