import { useState, useEffect, useCallback } from 'react'
import { CaretLeft, CaretRight, Play, Pause, ArrowCounterClockwise } from '@phosphor-icons/react'

interface Step {
  title: string
  description: string
  formula?: string
  whyItMatters?: string
  content: React.ReactNode
}

interface StepByStepProps {
  steps: Step[]
  title?: string
  autoplaySpeed?: number
}

export default function StepByStep({
  steps,
  title,
  autoplaySpeed = 3000,
}: StepByStepProps) {
  const [currentStep, setCurrentStep] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const goNext = useCallback(() => {
    setCurrentStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev))
  }, [steps.length])

  const goPrev = () => {
    setCurrentStep((prev) => (prev > 0 ? prev - 1 : prev))
  }

  const reset = () => {
    setCurrentStep(0)
    setIsPlaying(false)
  }

  useEffect(() => {
    if (!isPlaying) return
    if (currentStep >= steps.length - 1) {
      setIsPlaying(false)
      return
    }
    const timer = setTimeout(goNext, autoplaySpeed)
    return () => clearTimeout(timer)
  }, [isPlaying, currentStep, autoplaySpeed, goNext, steps.length])

  const step = steps[currentStep]
  const progress = ((currentStep + 1) / steps.length) * 100

  return (
    <div className="glass-card p-6 space-y-4">
      {title && <h3 className="text-lg font-semibold text-gray-900">{title}</h3>}

      {/* Progress bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-600">
          <span>
            Passo {currentStep + 1} de {steps.length}
          </span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        {/* Step dots */}
        <div className="flex gap-1 justify-center">
          {steps.map((_, i) => (
            <button
              key={i}
              onClick={() => setCurrentStep(i)}
              className={`w-2 h-2 rounded-full transition-all ${
                i === currentStep
                  ? 'bg-blue-400 scale-125'
                  : i < currentStep
                  ? 'bg-blue-600'
                  : 'bg-gray-300'
              }`}
            />
          ))}
        </div>
      </div>

      {/* Step content */}
      <div className="space-y-3">
        <h4 className="text-base font-semibold text-blue-600">{step.title}</h4>
        <p className="text-sm text-gray-700 leading-relaxed">{step.description}</p>
        <div>{step.content}</div>
        {step.whyItMatters && (
          <div className="border border-amber-200 bg-amber-50 rounded-sm px-4 py-3 text-sm text-amber-700">
            <span className="font-semibold text-amber-800">Na pratica: </span>
            {step.whyItMatters}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-200">
        <button
          onClick={goPrev}
          disabled={currentStep === 0}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          <CaretLeft weight="bold" className="w-4 h-4" /> Anterior
        </button>

        <div className="flex gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-2 rounded-sm bg-blue-600 hover:bg-blue-500 transition-colors"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          <button
            onClick={reset}
            className="p-2 rounded-sm bg-gray-100 hover:bg-gray-200 transition-colors"
          >
            <ArrowCounterClockwise className="w-4 h-4" />
          </button>
        </div>

        <button
          onClick={goNext}
          disabled={currentStep === steps.length - 1}
          className="flex items-center gap-1 px-3 py-1.5 text-sm rounded-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          Proximo <CaretRight weight="bold" className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
