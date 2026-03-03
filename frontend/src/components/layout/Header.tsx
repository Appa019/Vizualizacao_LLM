import { useLocation } from 'react-router-dom'

// ─── Mapa de títulos por rota ─────────────────────────────────────────────────

const titulosPorRota: Record<string, { titulo: string; subtitulo: string }> = {
  '/': {
    titulo: 'Início',
    subtitulo: 'Bem-vindo à plataforma educacional de LLMs',
  },
  '/what-are-llms': {
    titulo: 'O que são LLMs?',
    subtitulo: 'Grandes Modelos de Linguagem - conceitos e fundamentos',
  },
  '/tokenization': {
    titulo: 'Tokenização',
    subtitulo: 'Como texto se transforma em sequências numéricas',
  },
  '/embeddings': {
    titulo: 'Embeddings',
    subtitulo: 'Representações vetoriais de palavras e contextos',
  },
  '/architecture': {
    titulo: 'Arquitetura Transformer',
    subtitulo: 'A estrutura interna do modelo que mudou a IA',
  },
  '/attention': {
    titulo: 'Mecanismo de Atenção',
    subtitulo: 'Self-attention, Multi-head attention e Scaled Dot-Product',
  },
  '/training': {
    titulo: 'Treinamento',
    subtitulo: 'Pré-treino, ajuste fino e o processo de aprendizado',
  },
  '/inference': {
    titulo: 'Inferência',
    subtitulo: 'Geração de tokens e estratégias de decodificação',
  },
  '/fine-tuning': {
    titulo: 'Fine-tuning',
    subtitulo: 'Especialização de modelos para tarefas específicas',
  },
  '/lab': {
    titulo: 'Laboratório',
    subtitulo: 'Experimentos interativos com simulacoes',
  },
}

// ─── Componente principal ─────────────────────────────────────────────────────

export default function Header() {
  const location = useLocation()
  const info = titulosPorRota[location.pathname] ?? {
    titulo: 'LLM Explorer',
    subtitulo: '',
  }

  return (
    <header className="
      sticky top-0 z-20 flex items-center justify-between
      px-6 py-3.5 bg-white/90 backdrop-blur-xl
      border-b border-gray-200
    ">
      {/* Título da página atual */}
      <div className="min-w-0">
        <h1 className="text-base font-semibold text-gray-900 leading-tight truncate">
          {info.titulo}
        </h1>
        {info.subtitulo && (
          <p className="text-xs text-gray-500 leading-tight mt-0.5 truncate hidden sm:block">
            {info.subtitulo}
          </p>
        )}
      </div>
    </header>
  )
}
