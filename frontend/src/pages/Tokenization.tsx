import { useState, useEffect } from 'react'
import { Hash } from '@phosphor-icons/react'
import { useTokenize, useBPESteps, useCompareTokenizers } from '../api/hooks'
import StepByStep from '../components/education/StepByStep'
import FormulaBlock from '../components/education/FormulaBlock'
import WhyItMatters from '../components/education/WhyItMatters'
import EducationalViz from '../components/education/EducationalViz'
import ApiLoadingState from '../components/education/ApiLoadingState'

// ─── Paleta de cores para tokens ─────────────────────────────────────────────

const CORES_TOKENS = [
  'bg-blue-50 text-blue-700 border-blue-200',
  'bg-violet-50 text-violet-700 border-violet-200',
  'bg-emerald-50 text-emerald-700 border-emerald-200',
  'bg-orange-50 text-orange-700 border-orange-200',
  'bg-pink-50 text-pink-700 border-pink-200',
  'bg-yellow-50 text-yellow-700 border-yellow-200',
  'bg-cyan-50 text-cyan-700 border-cyan-200',
  'bg-red-50 text-red-700 border-red-200',
]

// ─── Dados fallback quando o backend esta offline ────────────────────────────

const FALLBACK_TOKENS = ['O', 'gato', 'dorm', 'iu', 'no', 'tap', 'ete']

const FALLBACK_BPE_STEPS = [
  { passo: 1, par: ['t', 'a'] as [string, string], frequencia: 3, tokens_atuais: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u', ' ', 'n', 'o', ' ', 'ta', 'p', 'e', 't', 'e'], vocabulario_tamanho: 21 },
  { passo: 2, par: ['n', 'o'] as [string, string], frequencia: 2, tokens_atuais: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u', ' ', 'no', ' ', 'ta', 'p', 'e', 't', 'e'], vocabulario_tamanho: 20 },
  { passo: 3, par: ['tap', 'e'] as [string, string], frequencia: 1, tokens_atuais: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u', ' ', 'no', ' ', 'tape', 't', 'e'], vocabulario_tamanho: 19 },
]

const FALLBACK_COMPARE = [
  { nome: 'Palavra', tokens: ['O', 'gato', 'dormiu', 'no', 'tapete'], num_tokens: 5, descricao: 'Divide por espacos e pontuacao' },
  { nome: 'Subpalavra (BPE)', tokens: ['O', 'gato', 'dorm', 'iu', 'no', 'tap', 'ete'], num_tokens: 7, descricao: 'Divide em subpalavras frequentes' },
  { nome: 'Caractere', tokens: ['O', ' ', 'g', 'a', 't', 'o', ' ', 'd', 'o', 'r', 'm', 'i', 'u'], num_tokens: 22, descricao: 'Cada caractere e um token' },
]

// ─── Componente principal ────────────────────────────────────────────────────

export default function Tokenization() {
  const [entrada, setEntrada] = useState('O gato dormiu no tapete')

  const tokenize = useTokenize()
  const bpeSteps = useBPESteps()
  const compareTokenizers = useCompareTokenizers()

  // Tokenizar ao mudar texto
  useEffect(() => {
    if (entrada.trim()) {
      tokenize.execute({ texto: entrada })
      bpeSteps.execute({ texto: entrada, num_merges: 10 })
    }
  }, [entrada])

  // Dados para exibicao
  const tokens = tokenize.data?.tokens ?? FALLBACK_TOKENS
  const contagem = tokenize.data?.num_tokens ?? tokens.length

  // BPE Steps para o StepByStep
  const bpeData = bpeSteps.data
  const bpeStepItems = bpeData
    ? bpeData.passos.map((passo) => ({
        title: `Passo ${passo.passo}: Fundir "${passo.par_mesclado[0]}" + "${passo.par_mesclado[1]}"`,
        description: `Par mais frequente com ${passo.frequencia} ocorrencia(s). Vocabulario: ${passo.tamanho_vocabulario} tokens. Novo token: "${passo.novo_token}".`,
        content: (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1">
              {passo.amostra_corpus.flat().map((t, i) => (
                <span
                  key={i}
                  className={`px-1.5 py-0.5 rounded border text-[11px] font-mono ${
                    t === passo.novo_token
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                      : 'bg-gray-100 border-gray-200 text-gray-600'
                  }`}
                >
                  {t === ' ' ? '\u00B7' : t}
                </span>
              ))}
            </div>
          </div>
        ),
      }))
    : FALLBACK_BPE_STEPS.map((passo) => ({
        title: `Passo ${passo.passo}: Fundir "${passo.par[0]}" + "${passo.par[1]}"`,
        description: `Par mais frequente com ${passo.frequencia} ocorrencia(s). Vocabulario: ${passo.vocabulario_tamanho} tokens.`,
        content: (
          <div className="flex flex-wrap gap-1">
            {passo.tokens_atuais.map((t, i) => (
              <span
                key={i}
                className="px-1.5 py-0.5 rounded border bg-gray-100 border-gray-200 text-[11px] font-mono text-gray-600"
              >
                {t === ' ' ? '\u00B7' : t}
              </span>
            ))}
          </div>
        ),
      }))

  // Dados do comparador
  const compareData = compareTokenizers.data?.comparacao ?? FALLBACK_COMPARE

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-10 animate-slide-up">
      {/* Header */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <Hash size={12} weight="regular" />
          Modulo 03
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Tokenizacao
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          Modelos de linguagem nao leem palavras - leem <em>tokens</em>. Tokens sao fragmentos
          de texto que podem ser palavras completas, partes de palavras ou ate caracteres
          individuais, dependendo do vocabulario do modelo.
        </p>
      </section>

      {/* Formula BPE */}
      <FormulaBlock
        formula={"\\text{merge}(t_a, t_b) \\to t_{ab} \\quad \\text{onde} \\quad (t_a, t_b) = \\arg\\max_{(x,y)} \\text{freq}(x, y)"}
        variables={[
          { symbol: 't_a', color: '#3b82f6', label: 'Token A', description: 'Primeiro token do par mais frequente' },
          { symbol: 't_b', color: '#22c55e', label: 'Token B', description: 'Segundo token do par mais frequente' },
          { symbol: 't_{ab}', color: '#a855f7', label: 'Token fundido', description: 'Novo token criado pela fusao de A e B' },
        ]}
        size="lg"
      />

      {/* Tokenizador interativo */}
      <EducationalViz
        title="Tokenizador Interativo"
        icon={<Hash size={18} weight="regular" />}
        caption="Digite qualquer texto e veja como ele e dividido em tokens e convertido para IDs numericos."
      >
        <div className="space-y-4">
          {/* Campo de entrada */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-1 block">
              Texto de entrada
            </label>
            <p className="text-[11px] text-gray-400 mb-2">Exemplo fixo para demonstracao</p>
            <textarea
              value={entrada}
              readOnly
              rows={3}
              className="w-full bg-gray-50 border border-gray-200 rounded-sm px-4 py-2.5 text-sm text-gray-900 font-mono resize-none cursor-default"
            />
          </div>

          {/* Contagem */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500 font-mono">{contagem} tokens</span>
            {tokenize.loading && (
              <span className="text-xs text-gray-400 animate-pulse">Tokenizando...</span>
            )}
          </div>

          {/* Tokens coloridos */}
          <ApiLoadingState
            loading={false}
            error={tokenize.error}
            fallback={
              <div className="flex flex-wrap gap-1.5 min-h-[40px]">
                {FALLBACK_TOKENS.map((t, i) => (
                  <span
                    key={i}
                    className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-mono ${CORES_TOKENS[i % CORES_TOKENS.length]} transition-all duration-150`}
                  >
                    {t}
                  </span>
                ))}
              </div>
            }
          >
            <div className="flex flex-wrap gap-1.5 min-h-[40px]">
              {tokens.map((t, i) => (
                <span
                  key={i}
                  className={`inline-flex items-center px-2 py-0.5 rounded border text-xs font-mono ${CORES_TOKENS[i % CORES_TOKENS.length]} transition-all duration-150`}
                >
                  {t === ' ' ? '\u00B7' : t}
                </span>
              ))}
              {tokens.length === 0 && (
                <span className="text-xs text-gray-400">
                  Digite texto acima para ver os tokens
                </span>
              )}
            </div>
          </ApiLoadingState>

          {/* Tabela Token -> Indice */}
          {tokens.length > 0 && (
            <div className="border border-gray-200 rounded-sm overflow-hidden">
              <div className="grid grid-cols-2 bg-gray-50 px-4 py-2 border-b border-gray-200">
                <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Indice</span>
                <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">Token</span>
              </div>
              <div className="max-h-48 overflow-y-auto">
                {tokens.slice(0, 20).map((t, i) => (
                  <div
                    key={i}
                    className="grid grid-cols-2 px-4 py-2 border-b border-gray-200 last:border-0 hover:bg-gray-50 transition-colors"
                  >
                    <span className="text-xs font-mono text-gray-400 tabular-nums">{i}</span>
                    <span className={`text-xs font-mono ${CORES_TOKENS[i % CORES_TOKENS.length].split(' ')[1]}`}>
                      "{t === ' ' ? '\u00B7' : t}"
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </EducationalViz>

      {/* Por que subpalavras? */}
      <section className="glass-card p-5">
        <h4 className="text-sm font-semibold text-gray-800 mb-2">Por que subpalavras?</h4>
        <p className="text-xs text-gray-600 leading-relaxed">
          Se usassemos palavras inteiras, o vocabulario precisaria conter todas as palavras
          possiveis - incluindo conjugacoes, plurais e neologismos. Isso e inviavel. Se usassemos
          caracteres, cada token teria pouca informacao semantica e as sequencias ficariam muito longas.
        </p>
        <p className="text-xs text-gray-600 leading-relaxed mt-2">
          Subpalavras (como BPE) sao o meio-termo ideal: um vocabulario compacto (~32k-100k tokens)
          que consegue representar qualquer texto, mantendo palavras comuns inteiras e dividindo
          palavras raras em pedacos reconheciveis.
        </p>
      </section>

      {/* BPE animado */}
      <div className="space-y-3">
        <ApiLoadingState
          loading={bpeSteps.loading}
          error={bpeSteps.error}
          compact
          fallback={
            <StepByStep
              steps={bpeStepItems}
              title="BPE - Byte Pair Encoding (dados de exemplo)"
              autoplaySpeed={3000}
            />
          }
        >
          {bpeStepItems.length > 0 && (
            <StepByStep
              steps={bpeStepItems}
              title="BPE - Byte Pair Encoding"
              autoplaySpeed={3000}
            />
          )}
        </ApiLoadingState>

        {bpeData && bpeData.passos.length > 0 && (
          <div className="grid grid-cols-2 gap-3">
            <div className="text-center p-3 rounded-sm bg-gray-100">
              <p className="text-xs text-gray-500">Mesclagens realizadas</p>
              <p className="text-lg font-mono text-blue-600">{bpeData.num_passos}</p>
            </div>
            <div className="text-center p-3 rounded-sm bg-gray-100">
              <p className="text-xs text-gray-500">Vocabulario final</p>
              <p className="text-lg font-mono text-green-600">{bpeData.passos[bpeData.passos.length - 1].tamanho_vocabulario}</p>
            </div>
          </div>
        )}
      </div>

      {/* Comparacao de tokenizadores */}
      <EducationalViz
        title="Comparacao de Abordagens"
        icon={<Hash size={18} weight="regular" />}
        caption="Veja como o mesmo texto e tokenizado de formas diferentes: por palavra, subpalavra (BPE) e caractere."
      >
        <ApiLoadingState
          loading={compareTokenizers.loading}
          error={compareTokenizers.error}
          fallback={
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {FALLBACK_COMPARE.map((abordagem) => (
                <div key={abordagem.nome} className="glass-card p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-emerald-600">{abordagem.nome}</h4>
                    <span className="text-xs font-mono text-gray-500">{abordagem.num_tokens} tokens</span>
                  </div>
                  <p className="text-[11px] text-gray-500">{abordagem.descricao}</p>
                  <div className="flex flex-wrap gap-1">
                    {abordagem.tokens.map((t, i) => (
                      <span key={i} className={`px-1.5 py-0.5 rounded border text-[11px] font-mono ${CORES_TOKENS[i % CORES_TOKENS.length]}`}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {compareData.map((item) => (
              <div key={'abordagem' in item ? item.abordagem : item.nome} className="glass-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold text-emerald-600">{'abordagem' in item ? item.abordagem : item.nome}</h4>
                  <span className="text-xs font-mono text-gray-500">{item.num_tokens} tokens</span>
                </div>
                <p className="text-[11px] text-gray-500">{item.descricao}</p>
                <div className="flex flex-wrap gap-1">
                  {item.tokens.map((t, i) => (
                    <span key={i} className={`px-1.5 py-0.5 rounded border text-[11px] font-mono ${CORES_TOKENS[i % CORES_TOKENS.length]}`}>
                      {t === ' ' ? '\u00B7' : t}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ApiLoadingState>
      </EducationalViz>

      {/* Conceitos importantes */}
      <section>
        <h3 className="text-base font-semibold text-gray-800 mb-4">Conceitos importantes</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {[
            {
              termo: 'BPE - Byte Pair Encoding',
              desc: 'O algoritmo mais comum. Comeca com caracteres individuais e iterativamente funde os pares mais frequentes ate atingir o tamanho de vocabulario desejado.',
            },
            {
              termo: 'Vocabulario',
              desc: 'GPT-4 usa ~100k tokens. Llama usa ~32k. Tamanho maior = menos tokens por texto, mas mais parametros no embedding.',
            },
            {
              termo: 'Tokens especiais',
              desc: '<|endoftext|>, <s>, </s> - tokens que o modelo usa para sinalizar inicio, fim e separacao de sequencias.',
            },
            {
              termo: 'Custo em tokens',
              desc: 'APIs de LLM cobram por token. Uma palavra em ingles ~= 1 token. Em portugues, pode ser 1.2-1.5 tokens.',
            },
          ].map((item) => (
            <div key={item.termo} className="glass-card p-4">
              <p className="text-xs font-semibold text-emerald-600 font-mono mb-2">{item.termo}</p>
              <p className="text-xs text-gray-500 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Por que importa */}
      <WhyItMatters>
        <p>
          A tokenizacao e o primeiro passo de todo pipeline de LLM - e afeta diretamente custo,
          qualidade e velocidade. Um tokenizador ruim gera mais tokens (mais caro e lento),
          enquanto um bom tokenizador comprime o texto eficientemente.
        </p>
        <p className="mt-2">
          Em portugues, tokenizadores treinados em ingles fragmentam palavras desnecessariamente:
          "computacao" pode virar ["comp", "uta", "cao"] (3 tokens) em vez de ["computacao"] (1 token).
          Por isso modelos multilingues como Llama 3 e GPT-4 usam vocabularios maiores.
        </p>
      </WhyItMatters>
    </div>
  )
}
