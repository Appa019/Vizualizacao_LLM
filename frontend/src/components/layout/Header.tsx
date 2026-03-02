import { useLocation } from 'react-router-dom'
import { Brain, Cpu, ToggleLeft, ToggleRight } from '@phosphor-icons/react'
import { useState } from 'react'

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
    subtitulo: 'Experimentos interativos com modelos reais',
  },
}

// ─── Props ────────────────────────────────────────────────────────────────────

interface HeaderProps {
  modoSimulacao: boolean
  onAlternarModo: (ativo: boolean) => void
}

// ─── Componente principal ─────────────────────────────────────────────────────

export default function Header({ modoSimulacao, onAlternarModo }: HeaderProps) {
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

      {/* Controles do header */}
      <div className="flex items-center gap-3 flex-shrink-0 ml-4">
        {/* Toggle simulação vs modelo real */}
        <button
          onClick={() => onAlternarModo(!modoSimulacao)}
          className={`
            group flex items-center gap-2.5 px-3 py-1.5 rounded-sm text-xs font-medium
            border transition-all duration-200
            ${
              modoSimulacao
                ? 'bg-violet-50 border-violet-200 text-violet-700 hover:bg-violet-100'
                : 'bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100'
            }
          `}
          title={modoSimulacao ? 'Usar modelo real (requer backend)' : 'Usar simulação local'}
        >
          {modoSimulacao ? (
            <>
              <Brain size={14} weight="duotone" className="text-violet-600" />
              <span className="hidden sm:inline">Simulação</span>
              <ToggleLeft size={16} weight="duotone" className="text-violet-600" />
            </>
          ) : (
            <>
              <Cpu size={14} weight="duotone" className="text-blue-600" />
              <span className="hidden sm:inline">Modelo Real</span>
              <ToggleRight size={16} weight="duotone" className="text-blue-600" />
            </>
          )}
        </button>

        {/* Indicador de status do backend */}
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-sm bg-gray-50 border border-gray-200">
          <span
            className={`
              w-1.5 h-1.5 rounded-full flex-shrink-0
              ${modoSimulacao ? 'bg-violet-400' : 'bg-green-400 animate-pulse'}
            `}
          />
          <span className="text-xs text-gray-500 hidden sm:inline">
            {modoSimulacao ? 'Local' : 'API'}
          </span>
        </div>
      </div>
    </header>
  )
}
