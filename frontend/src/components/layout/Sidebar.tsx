import { useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
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
  CaretLeft,
  CaretRight,
  BookOpenText,
  GearSix,
} from '@phosphor-icons/react'
import type { Icon } from '@phosphor-icons/react'

// ─── Definição dos módulos de navegação ──────────────────────────────────────

interface Modulo {
  caminho: string
  rotulo: string
  icone: Icon
  descricao: string
  numero: number
}

const modulos: Modulo[] = [
  {
    numero: 0,
    caminho: '/setup',
    rotulo: 'Configurar',
    icone: GearSix,
    descricao: 'Configurar modelo local',
  },
  {
    numero: 1,
    caminho: '/',
    rotulo: 'Início',
    icone: BookOpenText,
    descricao: 'Visão geral da plataforma',
  },
  {
    numero: 2,
    caminho: '/what-are-llms',
    rotulo: 'O que são LLMs?',
    icone: Brain,
    descricao: 'Conceitos fundamentais',
  },
  {
    numero: 3,
    caminho: '/tokenization',
    rotulo: 'Tokenização',
    icone: Hash,
    descricao: 'Como o texto vira números',
  },
  {
    numero: 4,
    caminho: '/embeddings',
    rotulo: 'Embeddings',
    icone: Stack,
    descricao: 'Representações vetoriais',
  },
  {
    numero: 5,
    caminho: '/architecture',
    rotulo: 'Arquitetura',
    icone: Cpu,
    descricao: 'Transformer por dentro',
  },
  {
    numero: 6,
    caminho: '/attention',
    rotulo: 'Atenção',
    icone: Eye,
    descricao: 'Mecanismo de self-attention',
  },
  {
    numero: 7,
    caminho: '/training',
    rotulo: 'Treinamento',
    icone: Student,
    descricao: 'Pré-treino e ajuste fino',
  },
  {
    numero: 8,
    caminho: '/inference',
    rotulo: 'Inferência',
    icone: Lightning,
    descricao: 'Geração de tokens',
  },
  {
    numero: 9,
    caminho: '/fine-tuning',
    rotulo: 'Fine-tuning',
    icone: Sliders,
    descricao: 'Especialização do modelo',
  },
  {
    numero: 10,
    caminho: '/lab',
    rotulo: 'Laboratório',
    icone: Flask,
    descricao: 'Experimentos interativos',
  },
]

// ─── Componente principal ─────────────────────────────────────────────────────

export default function Sidebar() {
  const [recolhido, setRecolhido] = useState(false)
  const location = useLocation()

  // Calcula progresso com base nos módulos visitados (simplificado)
  const moduloAtualIndex = modulos.findIndex((m) => m.caminho === location.pathname)
  const progressoPct = moduloAtualIndex > 0 ? Math.round((moduloAtualIndex / (modulos.length - 1)) * 100) : 0

  return (
    <aside
      className={`
        relative flex flex-col h-screen bg-gray-50 border-r border-gray-200
        transition-all duration-300 ease-in-out flex-shrink-0
        ${recolhido ? 'w-16' : 'w-64'}
      `}
    >
      {/* Botão de recolher */}
      <button
        onClick={() => setRecolhido(!recolhido)}
        className="
          absolute -right-3 top-6 z-10
          flex items-center justify-center w-6 h-6
          bg-white border border-gray-300 rounded-full
          text-gray-600 hover:text-gray-800 hover:border-gray-400
          transition-all duration-150 shadow-sm
        "
        aria-label={recolhido ? 'Expandir sidebar' : 'Recolher sidebar'}
      >
        {recolhido ? <CaretRight size={12} weight="bold" /> : <CaretLeft size={12} weight="bold" />}
      </button>

      {/* Logo e título */}
      <div className={`flex items-center gap-3 px-4 py-5 ${recolhido ? 'justify-center' : ''}`}>
        <div className="flex items-center justify-center w-8 h-8 bg-gray-900 rounded-sm flex-shrink-0">
          <span className="text-white font-mono text-xs font-bold">LX</span>
        </div>
        {!recolhido && (
          <div className="min-w-0">
            <p className="text-sm font-semibold text-gray-900 leading-tight truncate">LLM Explorer</p>
            <p className="text-xs text-gray-500 leading-tight truncate">Plataforma educacional</p>
          </div>
        )}
      </div>

      <div className="mx-4 divider-subtle mb-3" />

      {/* Barra de progresso */}
      {!recolhido && (
        <div className="px-4 mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-gray-500 font-medium">Progresso</span>
            <span className="text-xs text-blue-400 font-medium tabular-nums">{progressoPct}%</span>
          </div>
          <div className="h-1 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progressoPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Lista de navegação */}
      <nav className="flex-1 overflow-y-auto px-2 pb-4 space-y-0.5">
        {modulos.map((modulo) => {
          const Icone = modulo.icone
          const ativo = location.pathname === modulo.caminho

          return (
            <div key={modulo.caminho}>
              <NavLink
                to={modulo.caminho}
                title={recolhido ? modulo.rotulo : undefined}
                className={`
                  group flex items-center gap-3 rounded-sm px-2.5 py-2
                  transition-all duration-150 relative
                  ${recolhido ? 'justify-center' : ''}
                  ${
                    ativo
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 border border-transparent'
                  }
                `}
              >
                {ativo && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r-full bg-blue-500" />
                )}

                {!recolhido && (
                  <span
                    className={`
                      flex-shrink-0 w-5 h-5 rounded text-center leading-5
                      text-[10px] font-semibold tabular-nums
                      ${ativo ? 'bg-blue-100 text-blue-700' : 'bg-gray-200 text-gray-500 group-hover:text-gray-600'}
                    `}
                  >
                    {modulo.numero}
                  </span>
                )}

                <Icone
                  size={16}
                  weight={ativo ? 'fill' : 'duotone'}
                  className={`flex-shrink-0 transition-transform duration-150 group-hover:scale-110 ${
                    recolhido ? 'mx-auto' : ''
                  }`}
                />

                {!recolhido && (
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium leading-tight truncate">{modulo.rotulo}</p>
                    <p
                      className={`text-[11px] leading-tight truncate mt-0.5 transition-colors ${
                        ativo
                          ? 'text-blue-600'
                          : 'text-gray-400 group-hover:text-gray-500'
                      }`}
                    >
                      {modulo.descricao}
                    </p>
                  </div>
                )}
              </NavLink>
            </div>
          )
        })}
      </nav>

      {/* Rodapé da sidebar */}
      {!recolhido && (
        <div className="px-4 py-4 border-t border-gray-200">
          <p className="text-[11px] text-gray-400 leading-relaxed">
            Aprenda como os grandes modelos de linguagem funcionam, passo a passo.
          </p>
        </div>
      )}
    </aside>
  )
}
