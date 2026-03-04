import { useRef, useState, useEffect } from 'react'
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
  GithubLogo,
  ArrowRight,
  ArrowUpRight,
  Terminal,
  CopySimple,
  Check,
  Cube,
  Function,
  ChartBar,
  Atom,
} from '@phosphor-icons/react'

// ─── Dados dos modulos ─────────────────────────────────────────────────────────

const modulos = [
  {
    numero: '01',
    titulo: 'O que sao LLMs?',
    descricao: 'O que sao Grandes Modelos de Linguagem e por que transformaram a IA.',
    icone: Brain,
    cor: 'blue',
  },
  {
    numero: '02',
    titulo: 'Tokenizacao',
    descricao: 'Como texto bruto vira sequencias de tokens - BPE, WordPiece e comparativos.',
    icone: Hash,
    cor: 'emerald',
  },
  {
    numero: '03',
    titulo: 'Embeddings',
    descricao: 'Vetores de alta dimensao que codificam significado semantico em espaco geometrico.',
    icone: Stack,
    cor: 'violet',
  },
  {
    numero: '04',
    titulo: 'Arquitetura Transformer',
    descricao: 'Encoder-decoder, camadas de atencao empilhadas, feed-forward e residual connections.',
    icone: Cpu,
    cor: 'orange',
  },
  {
    numero: '05',
    titulo: 'Mecanismo de Atencao',
    descricao: 'Self-attention, queries, keys e values com visualizacao de heatmaps interativos.',
    icone: Eye,
    cor: 'red',
  },
  {
    numero: '06',
    titulo: 'Treinamento',
    descricao: 'Pre-treino em escala, backpropagation, funcao de perda e descida de gradiente.',
    icone: Student,
    cor: 'yellow',
  },
  {
    numero: '07',
    titulo: 'Inferencia',
    descricao: 'Geracao token a token, temperatura, top-k e estrategias de decodificacao.',
    icone: Lightning,
    cor: 'cyan',
  },
  {
    numero: '08',
    titulo: 'Fine-tuning',
    descricao: 'Especializacao de modelos base com LoRA, PEFT e datasets reduzidos.',
    icone: Sliders,
    cor: 'pink',
  },
  {
    numero: '09',
    titulo: 'Laboratorio',
    descricao: 'Experimentos interativos: tokenize texto, explore embeddings, observe atencao ao vivo.',
    icone: Flask,
    cor: 'teal',
  },
]

// Estilos por cor - seguindo o padrao de Home.tsx
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

// ─── Dados das screenshots ─────────────────────────────────────────────────────

const screenshots = [
  {
    arquivo: '/screenshots/home.png',
    titulo: 'Pagina inicial',
    descricao: '9 modulos de aprendizado sequencial',
    rotacao: '-1.5deg',
  },
  {
    arquivo: '/screenshots/what-are-llms.png',
    titulo: 'O que sao LLMs',
    descricao: 'Teoria com formulas KaTeX e exemplos',
    rotacao: '0.5deg',
  },
  {
    arquivo: '/screenshots/tokenization.png',
    titulo: 'Tokenizacao',
    descricao: 'BPE interativo com visualizacao de merges',
    rotacao: '-0.8deg',
  },
  {
    arquivo: '/screenshots/architecture.png',
    titulo: 'Arquitetura Transformer',
    descricao: 'Visualizacao 3D do fluxo de tokens',
    rotacao: '1.2deg',
  },
  {
    arquivo: '/screenshots/inference.png',
    titulo: 'Inferencia',
    descricao: 'Pipeline de geracao token a token',
    rotacao: '-0.6deg',
  },
  {
    arquivo: '/screenshots/fine-tuning.png',
    titulo: 'Fine-tuning e LoRA',
    descricao: 'Visualizacao de adapters e PEFT',
    rotacao: '1.0deg',
  },
]

// ─── Stack tecnologico ─────────────────────────────────────────────────────────

const tecnologias = [
  { nome: 'React 18', categoria: 'Frontend', icone: Atom },
  { nome: 'TypeScript', categoria: 'Frontend', icone: Function },
  { nome: 'Three.js', categoria: 'Visualizacao', icone: Cube },
  { nome: 'Plotly.js', categoria: 'Visualizacao', icone: ChartBar },
  { nome: 'KaTeX', categoria: 'Math', icone: Function },
  { nome: 'FastAPI', categoria: 'Backend', icone: Lightning },
  { nome: 'Vite', categoria: 'Build', icone: Lightning },
  { nome: 'Tailwind CSS', categoria: 'Estilos', icone: Stack },
]

// ─── Componente: Botao de copiar ───────────────────────────────────────────────

function BotaoCopiar({ texto }: { texto: string }) {
  const [copiado, setCopiado] = useState(false)

  function copiar() {
    navigator.clipboard.writeText(texto).then(() => {
      setCopiado(true)
      setTimeout(() => setCopiado(false), 2000)
    })
  }

  return (
    <button
      onClick={copiar}
      aria-label="Copiar comando"
      className="
        flex items-center gap-1.5 text-xs font-mono text-gray-400
        hover:text-gray-200 transition-colors duration-150
        px-2 py-1 rounded-sm hover:bg-white/10
      "
    >
      {copiado ? (
        <>
          <Check size={13} weight="bold" className="text-emerald-400" />
          <span className="text-emerald-400">Copiado</span>
        </>
      ) : (
        <>
          <CopySimple size={13} weight="duotone" />
          Copiar
        </>
      )}
    </button>
  )
}

// ─── Componente: Bloco de codigo ───────────────────────────────────────────────

function BlocoComando({ comentario, comando }: { comentario?: string; comando: string }) {
  return (
    <div className="group relative">
      {comentario && (
        <div className="font-mono text-xs text-gray-500 mb-1"># {comentario}</div>
      )}
      <div className="flex items-center justify-between bg-gray-800 border border-gray-700 rounded-sm px-4 py-2.5">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-gray-500 font-mono text-xs shrink-0">$</span>
          <code className="font-mono text-sm text-gray-100 truncate">{comando}</code>
        </div>
        <BotaoCopiar texto={comando} />
      </div>
    </div>
  )
}

// ─── Componente: Secao de instalacao ──────────────────────────────────────────

function SecaoInstalacao() {
  const passos = [
    {
      numero: '1',
      titulo: 'Clone o repositorio',
      comando: 'git clone https://github.com/Appa019/Vizualizacao_LLM.git',
      comentario: undefined,
    },
    {
      numero: '2',
      titulo: 'Entre na pasta do projeto',
      comando: 'cd Vizualizacao_LLM',
      comentario: undefined,
    },
    {
      numero: '3',
      titulo: 'Inicie backend e frontend',
      comando: './start.sh',
      comentario: 'Cria venv, instala deps e sobe ambos os servidores',
    },
  ]

  return (
    <div className="space-y-5">
      {passos.map((passo) => (
        <div key={passo.numero} className="flex gap-5">
          <div className="flex flex-col items-center gap-1 shrink-0">
            <div className="w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-mono font-bold flex items-center justify-center">
              {passo.numero}
            </div>
            {passo.numero !== '3' && (
              <div className="w-px flex-1 bg-gray-200 min-h-[1.5rem]" />
            )}
          </div>
          <div className="flex-1 pb-5">
            <div className="text-sm font-medium text-gray-900 mb-2">{passo.titulo}</div>
            <BlocoComando comentario={passo.comentario} comando={passo.comando} />
          </div>
        </div>
      ))}

      <div className="mt-2 ml-11 text-xs text-gray-500 leading-relaxed">
        Apos iniciar, acesse{' '}
        <code className="code-inline">localhost:5173</code> no navegador.
        Python 3.11+ e Node 18+ sao requeridos.
      </div>
    </div>
  )
}

// ─── Componente: Carrossel de screenshots ─────────────────────────────────────

function CarrosselScreenshots() {
  const trilhaRef = useRef<HTMLDivElement>(null)
  const [indiceAtivo, setIndiceAtivo] = useState(0)

  // Rolagem ao clicar nos indicadores
  function irPara(indice: number) {
    setIndiceAtivo(indice)
    const trilha = trilhaRef.current
    if (!trilha) return
    const filho = trilha.children[indice] as HTMLElement
    if (filho) {
      const offsetEsquerda = filho.offsetLeft - trilha.offsetLeft
      trilha.scrollTo({ left: offsetEsquerda - 24, behavior: 'smooth' })
    }
  }

  // Atualiza indice ativo com IntersectionObserver
  useEffect(() => {
    const trilha = trilhaRef.current
    if (!trilha) return

    const observador = new IntersectionObserver(
      (entradas) => {
        for (const entrada of entradas) {
          if (entrada.isIntersecting && entrada.intersectionRatio > 0.5) {
            const indice = Array.from(trilha.children).indexOf(entrada.target as HTMLElement)
            if (indice >= 0) setIndiceAtivo(indice)
          }
        }
      },
      { root: trilha, threshold: 0.5 }
    )

    Array.from(trilha.children).forEach((filho) => observador.observe(filho))
    return () => observador.disconnect()
  }, [])

  return (
    <div>
      {/* Trilha horizontal com overflow */}
      <div
        ref={trilhaRef}
        className="flex gap-5 overflow-x-auto pb-6 -mx-6 px-6"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        {screenshots.map((sc, i) => (
          <div
            key={sc.arquivo}
            className="shrink-0 w-[320px] sm:w-[400px] lg:w-[480px] cursor-pointer"
            onClick={() => irPara(i)}
            style={{ transform: `rotate(${sc.rotacao})`, transformOrigin: 'center bottom' }}
          >
            <div
              className="
                glass-card overflow-hidden
                transition-all duration-300
                hover:shadow-md hover:-translate-y-0.5
              "
            >
              {/* Screenshot */}
              <div className="bg-gray-100 border-b border-gray-200">
                {/* Barra de titulo do navegador (decorativa) */}
                <div className="flex items-center gap-1.5 px-3 py-2 border-b border-gray-200 bg-gray-50">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-300" />
                  <span className="w-2.5 h-2.5 rounded-full bg-yellow-300" />
                  <span className="w-2.5 h-2.5 rounded-full bg-green-300" />
                  <div className="flex-1 mx-3 bg-white border border-gray-200 rounded-sm px-2 py-0.5">
                    <span className="font-mono text-[10px] text-gray-400">localhost:5173</span>
                  </div>
                </div>
                <img
                  src={sc.arquivo}
                  alt={`Screenshot da pagina: ${sc.titulo}`}
                  className="w-full h-[200px] sm:h-[240px] object-cover object-top"
                  loading="lazy"
                />
              </div>
              {/* Legenda */}
              <div className="px-4 py-3">
                <div className="text-sm font-medium text-gray-900">{sc.titulo}</div>
                <div className="text-xs text-gray-500 mt-0.5">{sc.descricao}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Indicadores */}
      <div className="flex items-center justify-center gap-2 mt-2">
        {screenshots.map((_, i) => (
          <button
            key={i}
            onClick={() => irPara(i)}
            aria-label={`Ver screenshot ${i + 1}`}
            className={`
              h-1.5 rounded-full transition-all duration-200
              ${i === indiceAtivo ? 'w-6 bg-blue-600' : 'w-1.5 bg-gray-300 hover:bg-gray-400'}
            `}
          />
        ))}
      </div>
    </div>
  )
}

// ─── Componente principal: LandingPage ────────────────────────────────────────

export default function LandingPage() {
  const GITHUB_URL = 'https://github.com/Appa019/Vizualizacao_LLM'

  return (
    <div className="min-h-screen bg-white text-gray-900">

      {/* ── Barra de navegacao ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          {/* Logo / nome */}
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-blue-600 rounded-sm flex items-center justify-center">
              <Brain size={15} weight="fill" className="text-white" />
            </div>
            <span className="font-semibold text-gray-900 text-sm tracking-tight">
              LLM Explorer
            </span>
            <span className="hidden sm:block font-mono text-[10px] text-gray-400 border border-gray-200 px-1.5 py-0.5 rounded-sm">
              pt-BR
            </span>
          </div>

          {/* Links de navegacao */}
          <nav className="hidden md:flex items-center gap-6">
            <a
              href="#modulos"
              className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
            >
              Modulos
            </a>
            <a
              href="#screenshots"
              className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
            >
              Screenshots
            </a>
            <a
              href="#stack"
              className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
            >
              Stack
            </a>
            <a
              href="#instalacao"
              className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
            >
              Instalacao
            </a>
          </nav>

          {/* CTA GitHub */}
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 btn-outline text-sm py-1.5 px-3"
          >
            <GithubLogo size={15} weight="fill" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </header>

      <main>

        {/* ── Hero ───────────────────────────────────────────────────────────── */}
        <section className="max-w-6xl mx-auto px-6 pt-20 pb-24">
          <div className="max-w-3xl">
            {/* Badge de contexto */}
            <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-7">
              <Brain size={12} weight="duotone" />
              Plataforma Educacional - Open Source
            </div>

            {/* Titulo principal */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-[1.1] tracking-tight mb-6">
              Aprenda como
              <br />
              <span className="text-blue-600">LLMs funcionam</span>
              <br />
              de dentro para fora.
            </h1>

            <p className="text-lg text-gray-600 leading-relaxed max-w-2xl mb-10">
              Uma plataforma interativa em portugues para estudar Transformers e Grandes Modelos
              de Linguagem. Visualizacoes 3D, formulas matematicas renderizadas, e experimentos
              ao vivo - tudo sem precisar de GPU.
            </p>

            {/* CTAs */}
            <div className="flex flex-wrap items-center gap-3">
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 btn-primary text-sm py-2.5 px-5"
              >
                <GithubLogo size={16} weight="fill" />
                Ver no GitHub
                <ArrowUpRight size={14} weight="bold" className="ml-0.5" />
              </a>
              <a
                href="#instalacao"
                className="flex items-center gap-2 btn-outline text-sm py-2.5 px-5"
              >
                <Terminal size={16} weight="duotone" />
                Rodar localmente
              </a>
            </div>

            {/* Estatisticas rapidas */}
            <div className="flex flex-wrap items-center gap-6 mt-12 pt-8 border-t border-gray-200">
              {[
                { valor: '9', rotulo: 'modulos de conteudo' },
                { valor: '3D', rotulo: 'visualizacoes interativas' },
                { valor: 'pt-BR', rotulo: 'conteudo em portugues' },
                { valor: 'MIT', rotulo: 'licenca open source' },
              ].map((stat) => (
                <div key={stat.rotulo}>
                  <div className="font-mono text-xl font-bold text-gray-900">{stat.valor}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{stat.rotulo}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Modulos ────────────────────────────────────────────────────────── */}
        <section
          id="modulos"
          className="bg-gray-50 border-y border-gray-200 py-20"
        >
          <div className="max-w-6xl mx-auto px-6">
            {/* Cabecalho da secao */}
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-12">
              <div>
                <div className="font-mono text-[11px] uppercase tracking-widest text-gray-400 mb-3">
                  Conteudo
                </div>
                <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 leading-tight tracking-tight">
                  9 modulos progressivos
                </h2>
                <p className="text-gray-500 text-sm mt-2 max-w-md leading-relaxed">
                  Do conceito basico ate experimentos avancados de fine-tuning.
                  Cada modulo inclui teoria, visualizacao e pratica.
                </p>
              </div>
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors shrink-0"
              >
                Ver codigo-fonte
                <ArrowRight size={14} weight="bold" />
              </a>
            </div>

            {/* Grid de modulos */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {modulos.map((modulo) => {
                const Icone = modulo.icone
                return (
                  <div
                    key={modulo.numero}
                    className="
                      group flex flex-col glass-card p-5 gap-4
                      hover:border-gray-300 hover:shadow-sm hover:bg-white
                      transition-all duration-200
                    "
                  >
                    {/* Topo */}
                    <div className="flex items-start justify-between">
                      <div
                        className={`
                          flex items-center justify-center w-9 h-9
                          rounded-sm border ${coresBadge[modulo.cor]}
                        `}
                      >
                        <Icone size={18} weight="duotone" className={coresIcone[modulo.cor]} />
                      </div>
                      <span className="text-xs font-mono text-gray-300 tabular-nums">
                        {modulo.numero}
                      </span>
                    </div>

                    {/* Conteudo */}
                    <div className="flex-1">
                      <h3 className="text-sm font-semibold text-gray-900 mb-1.5">
                        {modulo.titulo}
                      </h3>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        {modulo.descricao}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </section>

        {/* ── Screenshots ────────────────────────────────────────────────────── */}
        <section id="screenshots" className="py-20">
          <div className="max-w-6xl mx-auto px-6">
            {/* Cabecalho da secao */}
            <div className="mb-12">
              <div className="font-mono text-[11px] uppercase tracking-widest text-gray-400 mb-3">
                Interface
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 leading-tight tracking-tight">
                Veja em acao
              </h2>
              <p className="text-gray-500 text-sm mt-2 max-w-md leading-relaxed">
                Cada pagina combina conteudo didatico com visualizacoes interativas.
                Arraste para explorar as telas.
              </p>
            </div>

            {/* Carrossel */}
            <CarrosselScreenshots />
          </div>
        </section>

        {/* ── Stack tecnologico ──────────────────────────────────────────────── */}
        <section
          id="stack"
          className="bg-gray-50 border-y border-gray-200 py-20"
        >
          <div className="max-w-6xl mx-auto px-6">
            {/* Cabecalho */}
            <div className="mb-12">
              <div className="font-mono text-[11px] uppercase tracking-widest text-gray-400 mb-3">
                Tecnologias
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 leading-tight tracking-tight">
                Stack moderna e pragmatica
              </h2>
              <p className="text-gray-500 text-sm mt-2 max-w-md leading-relaxed">
                Nenhuma dependencia pesada desnecessaria. Cada biblioteca foi escolhida
                para maximizar a expressividade educacional.
              </p>
            </div>

            {/* Tabela de tecnologias */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-px bg-gray-200 border border-gray-200 rounded-sm overflow-hidden">
              {tecnologias.map((tech) => {
                const Icone = tech.icone
                return (
                  <div
                    key={tech.nome}
                    className="bg-white px-5 py-4 flex items-center gap-4 hover:bg-gray-50 transition-colors duration-150"
                  >
                    <div className="w-8 h-8 bg-gray-100 border border-gray-200 rounded-sm flex items-center justify-center shrink-0">
                      <Icone size={16} weight="duotone" className="text-gray-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-semibold text-gray-900">{tech.nome}</div>
                      <div className="text-xs text-gray-500">{tech.categoria}</div>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Nota sobre o simulador */}
            <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-sm flex gap-3">
              <div className="w-1 bg-blue-500 rounded-full shrink-0" />
              <p className="text-xs text-blue-700 leading-relaxed">
                O backend FastAPI inclui um simulador de Transformer completamente offline
                (sem GPU) e suporte opcional a modelos reais via HuggingFace
                (requer hardware adequado).
              </p>
            </div>
          </div>
        </section>

        {/* ── Instalacao ─────────────────────────────────────────────────────── */}
        <section id="instalacao" className="py-20">
          <div className="max-w-6xl mx-auto px-6">
            <div className="grid lg:grid-cols-2 gap-16 items-start">
              {/* Coluna esquerda: contexto */}
              <div>
                <div className="font-mono text-[11px] uppercase tracking-widest text-gray-400 mb-3">
                  Instalacao
                </div>
                <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 leading-tight tracking-tight mb-4">
                  Rode em 3 comandos.
                </h2>
                <p className="text-gray-500 text-sm leading-relaxed mb-8">
                  O script <code className="code-inline">start.sh</code> automatiza a criacao
                  do ambiente virtual Python, instalacao de dependencias e inicializacao dos
                  dois servidores em paralelo.
                </p>

                {/* Requisitos */}
                <div className="space-y-2">
                  <div className="text-xs font-mono uppercase tracking-widest text-gray-400 mb-3">
                    Requisitos
                  </div>
                  {[
                    { label: 'Python', versao: '3.11 ou superior' },
                    { label: 'Node.js', versao: '18 ou superior' },
                    { label: 'RAM', versao: '4 GB minimo' },
                    { label: 'GPU', versao: 'Opcional (simulador funciona sem)' },
                  ].map((req) => (
                    <div
                      key={req.label}
                      className="flex items-center justify-between py-2.5 border-b border-gray-100"
                    >
                      <span className="text-sm font-mono text-gray-600">{req.label}</span>
                      <span className="text-xs text-gray-500">{req.versao}</span>
                    </div>
                  ))}
                </div>

                <a
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-8 flex items-center gap-2 btn-primary text-sm py-2.5 px-5 w-fit"
                >
                  <GithubLogo size={16} weight="fill" />
                  Ver README completo
                  <ArrowUpRight size={14} weight="bold" className="ml-0.5" />
                </a>
              </div>

              {/* Coluna direita: comandos */}
              <div>
                <div className="glass-card p-6">
                  <div className="flex items-center gap-2 mb-5 pb-4 border-b border-gray-200">
                    <Terminal size={15} weight="duotone" className="text-gray-500" />
                    <span className="font-mono text-xs text-gray-500 uppercase tracking-widest">
                      Terminal
                    </span>
                  </div>
                  <SecaoInstalacao />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── CTA final ──────────────────────────────────────────────────────── */}
        <section className="bg-gray-900 py-20">
          <div className="max-w-6xl mx-auto px-6">
            <div className="max-w-2xl">
              <div className="font-mono text-[11px] uppercase tracking-widest text-gray-500 mb-5">
                Open Source
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white leading-tight tracking-tight mb-4">
                Contribua ou use como base
                <br />
                para seus proprios estudos.
              </h2>
              <p className="text-gray-400 text-sm leading-relaxed mb-8 max-w-lg">
                O projeto esta disponivel no GitHub com licenca MIT. Pull requests,
                issues e sugestoes de novos modulos sao bem-vindos.
              </p>
              <div className="flex flex-wrap gap-3">
                <a
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-white text-gray-900 hover:bg-gray-100 font-medium px-5 py-2.5 rounded-sm text-sm transition-colors duration-150"
                >
                  <GithubLogo size={16} weight="fill" />
                  Abrir no GitHub
                  <ArrowUpRight size={14} weight="bold" className="ml-0.5" />
                </a>
                <a
                  href={`${GITHUB_URL}/issues`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 bg-transparent border border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white font-medium px-5 py-2.5 rounded-sm text-sm transition-colors duration-150"
                >
                  Abrir uma Issue
                </a>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────────── */}
      <footer className="border-t border-gray-200 bg-white py-8">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 bg-blue-600 rounded-sm flex items-center justify-center">
              <Brain size={13} weight="fill" className="text-white" />
            </div>
            <span className="text-sm font-medium text-gray-900">LLM Explorer</span>
          </div>
          <div className="flex flex-wrap items-center gap-5 text-xs text-gray-400">
            <span>Plataforma educacional sobre LLMs e Transformers</span>
            <span className="hidden sm:block text-gray-200">|</span>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-gray-600 transition-colors"
            >
              GitHub
            </a>
            <span className="font-mono">MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
