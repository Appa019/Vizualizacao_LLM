import { useState } from 'react'
import { Info, BookOpenText, Calculator, Eye } from '@phosphor-icons/react'
import WhyItMatters from './WhyItMatters'
import FormulaBlock from './FormulaBlock'

type DetailLevel = 'simple' | 'detailed' | 'math'

interface FormulaConfig {
  formula: string
  variables?: {
    symbol: string
    color: string
    label: string
    description: string
  }[]
}

interface EducationalVizProps {
  title: string
  icon?: React.ReactNode
  caption: string
  formula?: FormulaConfig
  ocultarFormula?: boolean
  whyItMatters?: React.ReactNode
  children: React.ReactNode
  detailLevels?: {
    simple?: React.ReactNode
    detailed?: React.ReactNode
    math?: React.ReactNode
  }
}

export default function EducationalViz({
  title,
  icon,
  caption,
  formula,
  ocultarFormula,
  whyItMatters,
  children,
  detailLevels,
}: EducationalVizProps) {
  const [detailLevel, setDetailLevel] = useState<DetailLevel>('simple')
  const [highlightedVar, setHighlightedVar] = useState<string | null>(null)

  const levelLabels: Record<DetailLevel, { label: string; icon: React.ReactNode }> = {
    simple: { label: 'Simples', icon: <Eye weight="duotone" className="w-3.5 h-3.5" /> },
    detailed: { label: 'Detalhado', icon: <BookOpenText weight="duotone" className="w-3.5 h-3.5" /> },
    math: { label: 'Matematico', icon: <Calculator className="w-3.5 h-3.5" /> },
  }

  return (
    <div className="glass-card overflow-hidden">
      {/* Header */}
      <div className="px-6 pt-5 pb-3 border-b border-gray-200">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            {icon && (
              <div className="p-2 rounded-sm bg-blue-500/10 text-blue-400">{icon}</div>
            )}
            <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          </div>

          {/* Detail level toggle */}
          {detailLevels && (
            <div className="flex bg-gray-100 rounded-sm p-0.5 text-xs">
              {(Object.keys(levelLabels) as DetailLevel[]).map((level) => (
                <button
                  key={level}
                  onClick={() => setDetailLevel(level)}
                  className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md transition-all ${
                    detailLevel === level
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  {levelLabels[level].icon}
                  {levelLabels[level].label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Caption */}
        <div className="mt-3 flex items-start gap-2 text-sm text-gray-600 leading-relaxed">
          <Info weight="fill" className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-400" />
          <span>{caption}</span>
        </div>
      </div>

      {/* Visualization */}
      <div className="p-6">{children}</div>

      {/* Detail level content */}
      {detailLevels && detailLevels[detailLevel] && (
        <div className="px-6 pb-4 text-sm text-gray-700">
          {detailLevels[detailLevel]}
        </div>
      )}

      {/* Formula */}
      {formula && !ocultarFormula && (
        <div className="px-6 pb-4">
          <FormulaBlock
            formula={formula.formula}
            variables={formula.variables}
            highlightedVar={highlightedVar}
            onVariableClick={(sym) =>
              setHighlightedVar(highlightedVar === sym ? null : sym)
            }
          />
        </div>
      )}

      {/* Why it matters */}
      {whyItMatters && (
        <div className="px-6 pb-5">
          <WhyItMatters>{whyItMatters}</WhyItMatters>
        </div>
      )}
    </div>
  )
}
