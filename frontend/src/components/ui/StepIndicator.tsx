import { CheckFat } from '@phosphor-icons/react'

interface StepIndicatorProps {
  steps: string[]
  currentStep: number
  onStepClick?: (step: number) => void
}

export default function StepIndicator({
  steps,
  currentStep,
  onStepClick,
}: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-1">
      {steps.map((label, i) => (
        <div key={i} className="flex items-center">
          <button
            onClick={() => onStepClick?.(i)}
            disabled={!onStepClick}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-xs font-medium transition-all ${
              i === currentStep
                ? 'bg-blue-600 text-white'
                : i < currentStep
                ? 'bg-green-600/20 text-green-400 border border-green-500/30'
                : 'bg-gray-100 text-gray-500 border border-gray-200'
            } ${onStepClick ? 'cursor-pointer hover:brightness-110' : ''}`}
          >
            {i < currentStep ? (
              <CheckFat className="w-3 h-3" weight="fill" />
            ) : (
              <span className="w-4 text-center">{i + 1}</span>
            )}
            <span className="hidden sm:inline">{label}</span>
          </button>
          {i < steps.length - 1 && (
            <div
              className={`w-6 h-px mx-1 ${
                i < currentStep ? 'bg-green-500/50' : 'bg-gray-300'
              }`}
            />
          )}
        </div>
      ))}
    </div>
  )
}
