import { Link } from 'react-router-dom'
import {
  Brain,
  Hash,
  Stack,
  Cpu,
  Eye,
  Student,
  Lightning,
  Sliders,
  Flask,
  ArrowRight,
} from '@phosphor-icons/react'

// ─── Modulos de conteudo ──────────────────────────────────────────────────────

const modulos = [
  {
    numero: '01',
    caminho: '/what-are-llms',
    titulo: 'O que são LLMs?',
    descricao: 'Entenda o que são Grandes Modelos de Linguagem e por que mudaram o campo da IA.',
    icone: Brain,
    cor: 'blue',
  },
  {
    numero: '02',
    caminho: '/tokenization',
    titulo: 'Tokenização',
    descricao: 'Como texto bruto é convertido em sequências de tokens que o modelo processa.',
    icone: Hash,
    cor: 'emerald',
  },
  {
    numero: '03',
    caminho: '/embeddings',
    titulo: 'Embeddings',
    descricao: 'Vetores de alta dimensão que capturam significado e relações entre palavras.',
    icone: Stack,
    cor: 'violet',
  },
  {
    numero: '04',
    caminho: '/architecture',
    titulo: 'Arquitetura Transformer',
    descricao: 'A estrutura revolucionária de encoder-decoder que está por trás de todos os LLMs modernos.',
    icone: Cpu,
    cor: 'orange',
  },
  {
    numero: '05',
    caminho: '/attention',
    titulo: 'Mecanismo de Atenção',
    descricao: 'Self-attention, queries, keys e values - o coração do Transformer.',
    icone: Eye,
    cor: 'red',
  },
  {
    numero: '06',
    caminho: '/training',
    titulo: 'Treinamento',
    descricao: 'Pré-treino em escala, backpropagation e o processo de aprendizado.',
    icone: Student,
    cor: 'yellow',
  },
  {
    numero: '07',
    caminho: '/inference',
    titulo: 'Inferência',
    descricao: 'Geração de texto token a token, temperatura e estratégias de decodificação.',
    icone: Lightning,
    cor: 'cyan',
  },
  {
    numero: '08',
    caminho: '/fine-tuning',
    titulo: 'Fine-tuning',
    descricao: 'Como especializar um modelo base para tarefas específicas com poucos dados.',
    icone: Sliders,
    cor: 'pink',
  },
  {
    numero: '09',
    caminho: '/lab',
    titulo: 'Laboratório',
    descricao: 'Experimentos interativos - tokenize texto, visualize embeddings, observe a atenção.',
    icone: Flask,
    cor: 'teal',
  },
]

// Mapa de estilos por cor
const coresBadge: Record<string, string> = {
  blue: 'bg-blue-50 text-blue-700 border-blue-200',
  emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  violet: 'bg-violet-50 text-violet-700 border-violet-200',
  orange: 'bg-orange-50 text-orange-700 border-orange-200',
  red: 'bg-red-50 text-red-700 border-red-200',
  yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  cyan: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  pink: 'bg-pink-50 text-pink-700 border-pink-200',
  teal: 'bg-teal-50 text-teal-700 border-teal-200',
}

const coresIcone: Record<string, string> = {
  blue: 'text-blue-600',
  emerald: 'text-emerald-600',
  violet: 'text-violet-600',
  orange: 'text-orange-600',
  red: 'text-red-600',
  yellow: 'text-yellow-600',
  cyan: 'text-cyan-600',
  pink: 'text-pink-600',
  teal: 'text-teal-600',
}

// ─── Componente principal ─────────────────────────────────────────────────────

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {/* Cabecalho da pagina */}
      <div className="mb-12">
        <div className="inline-block font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          Plataforma educacional interativa
        </div>

        <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 leading-tight tracking-tight mb-4">
          Entenda LLMs de dentro<br className="hidden sm:block" /> para fora.
        </h2>

        <p className="text-base text-gray-600 leading-relaxed max-w-xl">
          Cada módulo combina teoria clara, visualizações interativas e conexões diretas
          com modelos reais. Do token ao texto gerado - tudo explicado em detalhe.
        </p>
      </div>

      {/* Grid de modulos - asymmetric 2-column */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {modulos.map((modulo) => {
          const Icone = modulo.icone
          return (
            <Link
              key={modulo.caminho}
              to={modulo.caminho}
              className="
                group flex flex-col glass-card p-5 gap-4
                hover:border-gray-300 hover:bg-gray-50
                transition-all duration-200
              "
            >
              {/* Topo do card */}
              <div className="flex items-start justify-between">
                <div className={`flex items-center justify-center w-9 h-9 rounded-sm border ${coresBadge[modulo.cor]}`}>
                  <Icone size={18} weight="duotone" className={coresIcone[modulo.cor]} />
                </div>
                <span className="text-xs font-mono text-gray-400 tabular-nums">{modulo.numero}</span>
              </div>

              {/* Conteudo */}
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-gray-900 mb-1.5 group-hover:text-gray-900 transition-colors">
                  {modulo.titulo}
                </h3>
                <p className="text-xs text-gray-500 leading-relaxed">
                  {modulo.descricao}
                </p>
              </div>

              {/* Link */}
              <div className="flex items-center gap-1 text-xs text-gray-400 group-hover:text-gray-600 transition-colors">
                <span>Explorar módulo</span>
                <ArrowRight
                  size={12}
                  weight="bold"
                  className="transition-transform duration-150 group-hover:translate-x-0.5"
                />
              </div>
            </Link>
          )
        })}
      </div>

      {/* Nota de rodape */}
      <div className="mt-10 pt-6 border-t border-gray-200">
        <p className="text-xs text-gray-400 text-center">
          Recomendado: siga os módulos em ordem para melhor compreensão.
          O Laboratório conecta todos os conceitos aprendidos.
        </p>
      </div>
    </div>
  )
}
