import { SpinnerGap, WifiX, Warning, ArrowClockwise } from '@phosphor-icons/react'

interface ApiLoadingStateProps {
  loading: boolean
  error: string | null
  children: React.ReactNode
  fallback?: React.ReactNode
  loadingMessage?: string
  compact?: boolean
}

export default function ApiLoadingState({
  loading,
  error,
  children,
  fallback,
  loadingMessage = 'Carregando dados do backend...',
  compact = false,
}: ApiLoadingStateProps) {
  if (loading) {
    return (
      <div
        className={`flex flex-col items-center justify-center gap-3 ${
          compact ? 'py-6' : 'py-12'
        }`}
      >
        <SpinnerGap weight="bold" className="w-6 h-6 text-blue-400 animate-spin" />
        <p className="text-sm text-gray-600">{loadingMessage}</p>
      </div>
    )
  }

  if (error) {
    if (fallback) {
      return (
        <div>
          <div className="flex items-center gap-2 px-3 py-2 mb-4 rounded-sm bg-amber-50 border border-amber-200 text-xs text-amber-700">
            <WifiX size={14} weight="regular" />
            <span>Backend offline - mostrando dados de exemplo</span>
          </div>
          {fallback}
        </div>
      )
    }

    return (
      <div
        className={`flex flex-col items-center justify-center gap-3 ${
          compact ? 'py-6' : 'py-12'
        }`}
      >
        <Warning weight="fill" className="w-6 h-6 text-red-400" />
        <p className="text-sm text-red-500 text-center max-w-md">{error}</p>
        <p className="text-xs text-gray-500 text-center">
          Verifique se o backend está rodando em localhost:8000
        </p>
      </div>
    )
  }

  return <>{children}</>
}

export function RetryButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
    >
      <ArrowClockwise size={12} weight="regular" />
      Tentar novamente
    </button>
  )
}
