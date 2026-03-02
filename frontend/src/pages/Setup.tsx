import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  GearSix,
  Cpu,
  Memory,
  Monitor,
  CheckCircle,
  XCircle,
  SpinnerGap,
  Warning,
  ArrowRight,
  ArrowCounterClockwise,
} from '@phosphor-icons/react'
import { useHardwareInfo, useAutoConfig } from '../api/hooks'

// ─── Componente de etapa no loading screen ──────────────────────────────────

function SetupStep({
  label,
  status,
  detalhe,
}: {
  label: string
  status: string
  detalhe: string
}) {
  const icon =
    status === 'completo' ? (
      <CheckCircle size={18} weight="fill" className="text-green-400" />
    ) : status === 'em_andamento' ? (
      <SpinnerGap size={18} weight="bold" className="text-blue-400 animate-spin" />
    ) : status === 'erro' ? (
      <XCircle size={18} weight="fill" className="text-red-400" />
    ) : (
      <div className="w-[18px] h-[18px] rounded-full border-2 border-gray-300" />
    )

  return (
    <div className="flex items-start gap-3 py-3">
      <div className="mt-0.5 flex-shrink-0">{icon}</div>
      <div className="min-w-0 flex-1">
        <p
          className={`text-sm font-medium ${
            status === 'completo'
              ? 'text-green-600'
              : status === 'em_andamento'
              ? 'text-blue-600'
              : status === 'erro'
              ? 'text-red-600'
              : 'text-gray-500'
          }`}
        >
          {label}
        </p>
        {detalhe && (
          <p className="text-xs text-gray-600 mt-0.5 break-words">{detalhe}</p>
        )}
      </div>
    </div>
  )
}

// ─── Componente principal ───────────────────────────────────────────────────

export default function Setup() {
  const { data: hardware, loading: hwLoading, error: hwError } = useHardwareInfo()
  const { data: configResult, loading: configuring, error: configError, execute: runConfig } = useAutoConfig()
  const [hasStarted, setHasStarted] = useState(false)

  const handleAutoConfig = () => {
    setHasStarted(true)
    runConfig(undefined as never)
  }

  const handleRetry = () => {
    setHasStarted(false)
    runConfig(undefined as never)
  }

  const isComplete = configResult?.sucesso === true
  const hasFailed = hasStarted && !configuring && configResult && !configResult.sucesso

  // Calcula progresso
  const etapas = configResult?.etapas ?? []
  const totalEtapas = etapas.length || 1
  const completedEtapas = etapas.filter((e) => e.status === 'completo').length
  const progressPct = Math.round((completedEtapas / totalEtapas) * 100)

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-8 animate-slide-up">
      {/* Header */}
      <section>
        <div className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-gray-500 bg-gray-100 border border-gray-200 px-2.5 py-1 rounded-sm mb-5">
          <GearSix size={12} weight="duotone" />
          Passo 0 - Configuracao
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 tracking-tight mb-4">
          Configurar Modelo Local
        </h2>
        <p className="text-gray-600 leading-relaxed max-w-2xl">
          Configure um modelo de linguagem real no seu hardware com um clique.
          O sistema detecta seus recursos e escolhe o modelo ideal automaticamente.
        </p>
      </section>

      {/* Hardware Info */}
      <section className="glass-card p-6">
        <h3 className="text-sm font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Monitor size={16} weight="duotone" className="text-blue-400" />
          Hardware Detectado
        </h3>

        {hwLoading ? (
          <div className="flex items-center gap-3 py-6 justify-center">
            <SpinnerGap className="w-5 h-5 text-blue-400 animate-spin" weight="bold" />
            <span className="text-sm text-gray-600">Detectando hardware...</span>
          </div>
        ) : hwError ? (
          <div className="flex items-center gap-2 py-4 text-sm text-red-700">
            <Warning size={16} weight="fill" />
            Backend offline - inicie o servidor para detectar o hardware
          </div>
        ) : hardware ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 p-3 rounded-sm bg-gray-100 border border-gray-200">
              <Cpu size={20} weight="duotone" className="text-blue-400 flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-xs text-gray-500">CPU</p>
                <p className="text-sm text-gray-800 truncate">{hardware.cpu}</p>
                <p className="text-xs text-gray-500">{hardware.nucleos} nucleos</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-sm bg-gray-100 border border-gray-200">
              <Memory size={20} weight="duotone" className="text-green-400 flex-shrink-0" />
              <div>
                <p className="text-xs text-gray-500">RAM</p>
                <p className="text-sm text-gray-800">{hardware.ram_total_gb} GB total</p>
                <p className="text-xs text-gray-500">{hardware.ram_disponivel_gb} GB disponivel</p>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-sm bg-gray-100 border border-gray-200">
              <Monitor size={20} weight="duotone" className={hardware.gpu_disponivel ? 'text-amber-400' : 'text-gray-400'} />
              <div>
                <p className="text-xs text-gray-500">GPU</p>
                <p className="text-sm text-gray-800">{hardware.gpu || 'Nao detectada'}</p>
                <p className="text-xs text-gray-500">
                  {hardware.gpu_disponivel ? 'Aceleracao disponivel' : 'Usando CPU'}
                </p>
              </div>
            </div>
          </div>
        ) : null}

        {/* Dependencias */}
        {hardware && (
          <div className="mt-4 flex flex-wrap gap-3">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-xs font-medium border ${
                hardware.torch_instalado
                  ? 'bg-green-50 border-green-200 text-green-700'
                  : 'bg-red-50 border-red-200 text-red-700'
              }`}
            >
              {hardware.torch_instalado ? <CheckCircle size={12} weight="fill" /> : <XCircle size={12} weight="fill" />}
              PyTorch {hardware.torch_version || 'nao instalado'}
            </span>
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-xs font-medium border ${
                hardware.transformers_instalado
                  ? 'bg-green-50 border-green-200 text-green-700'
                  : 'bg-red-50 border-red-200 text-red-700'
              }`}
            >
              {hardware.transformers_instalado ? <CheckCircle size={12} weight="fill" /> : <XCircle size={12} weight="fill" />}
              Transformers {hardware.transformers_version || 'nao instalado'}
            </span>
          </div>
        )}
      </section>

      {/* Loading Screen (apos clicar) */}
      {hasStarted && (configuring || configResult) ? (
        <section className="glass-card p-6">
          {/* Barra de progresso */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Progresso da configuracao</span>
              <span className="text-sm font-mono text-blue-400">{progressPct}%</span>
            </div>
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-700"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>

          {/* Etapas */}
          <div className="divide-y divide-gray-200">
            {etapas.length > 0
              ? etapas.map((etapa, i) => (
                  <SetupStep key={i} label={etapa.etapa} status={etapa.status} detalhe={etapa.detalhe} />
                ))
              : configuring && (
                  <SetupStep label="Iniciando configuracao..." status="em_andamento" detalhe="" />
                )}
          </div>

          {/* Erro geral */}
          {(configError || hasFailed) && (
            <div className="mt-4 p-4 rounded-sm bg-red-50 border border-red-200">
              <p className="text-sm text-red-700 mb-2">
                {configResult?.erro || configError || 'Erro durante a configuracao'}
              </p>
              {!hardware?.torch_instalado && (
                <div className="mt-2 p-3 bg-white rounded-sm">
                  <p className="text-xs text-gray-600 mb-1">Instale as dependencias:</p>
                  <code className="text-xs text-amber-600 font-mono">
                    pip install torch transformers
                  </code>
                </div>
              )}
              <button
                onClick={handleRetry}
                className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
              >
                <ArrowCounterClockwise size={14} weight="regular" /> Tentar Novamente
              </button>
            </div>
          )}

          {/* Sucesso */}
          {isComplete && (
            <div className="mt-6 p-5 rounded-sm bg-green-50 border border-green-200">
              <div className="flex items-center gap-3 mb-3">
                <CheckCircle className="w-6 h-6 text-green-500" weight="fill" />
                <h4 className="text-lg font-semibold text-green-700">
                  Modelo {configResult.modelo_nome} configurado!
                </h4>
              </div>
              {configResult.modelo_info && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Camadas', value: configResult.modelo_info.num_camadas },
                    { label: 'Cabecas', value: configResult.modelo_info.num_cabecas },
                    { label: 'd_model', value: configResult.modelo_info.d_model },
                    { label: 'Tipo', value: configResult.modelo_info.tipo },
                  ].map((item) => (
                    <div key={item.label} className="text-center p-2 rounded-sm bg-gray-100">
                      <p className="text-xs text-gray-500">{item.label}</p>
                      <p className="text-sm font-mono text-green-600">{item.value}</p>
                    </div>
                  ))}
                </div>
              )}
              <div className="flex flex-wrap gap-3">
                <Link
                  to="/what-are-llms"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
                >
                  Comecar Aprendizado <ArrowRight size={14} weight="bold" />
                </Link>
                <Link
                  to="/lab"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
                >
                  Ir para o Laboratorio <ArrowRight size={14} weight="bold" />
                </Link>
                <Link
                  to="/attention"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium transition-colors"
                >
                  Testar Atencao Real <ArrowRight size={14} weight="bold" />
                </Link>
              </div>
            </div>
          )}
        </section>
      ) : (
        /* Botao principal (antes de clicar) */
        <section className="flex flex-col items-center py-8">
          <button
            onClick={handleAutoConfig}
            disabled={hwLoading || !!hwError}
            className="group relative px-8 py-4 rounded-sm bg-blue-600 hover:bg-blue-500 disabled:bg-gray-300 text-white font-semibold text-lg transition-all duration-150"
          >
            <span className="flex items-center gap-3">
              <GearSix size={22} weight="duotone" className="group-hover:animate-spin" />
              Configurar Modelo Automaticamente
            </span>
          </button>
          <p className="mt-4 text-sm text-gray-500 text-center max-w-md">
            O sistema vai detectar seu hardware, escolher o melhor modelo
            e configura-lo automaticamente.
          </p>
        </section>
      )}
    </div>
  )
}
