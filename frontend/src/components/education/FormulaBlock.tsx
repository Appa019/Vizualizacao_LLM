import { useState } from 'react'
import 'katex/dist/katex.min.css'
import katex from 'katex'

interface Variable {
  symbol: string
  color: string
  label: string
  description: string
}

interface FormulaBlockProps {
  formula: string
  variables?: Variable[]
  onVariableClick?: (symbol: string) => void
  highlightedVar?: string | null
  size?: 'sm' | 'md' | 'lg'
}

export default function FormulaBlock({
  formula,
  variables = [],
  onVariableClick,
  highlightedVar,
  size = 'md',
}: FormulaBlockProps) {
  const [hoveredVar, setHoveredVar] = useState<string | null>(null)

  const sizeClasses = {
    sm: 'text-sm py-2 px-3',
    md: 'text-base py-3 px-4',
    lg: 'text-lg py-4 px-6',
  }

  const renderFormula = () => {
    try {
      let coloredFormula = formula
      variables.forEach((v) => {
        const isActive = hoveredVar === v.symbol || highlightedVar === v.symbol
        const color = isActive ? v.color : v.color
        const opacity = isActive ? '' : ''
        coloredFormula = coloredFormula.replace(
          new RegExp(`\\b${v.symbol.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g'),
          `\\textcolor{${color}}{\\mathbf{${v.symbol}}}`
        )
      })
      return katex.renderToString(coloredFormula, {
        throwOnError: false,
        displayMode: true,
      })
    } catch {
      return katex.renderToString(formula, {
        throwOnError: false,
        displayMode: true,
      })
    }
  }

  return (
    <div className="space-y-2">
      <div
        className={`glass-card ${sizeClasses[size]} overflow-x-auto`}
        dangerouslySetInnerHTML={{ __html: renderFormula() }}
      />

      {variables.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {variables.map((v) => (
            <button
              key={v.symbol}
              onMouseEnter={() => setHoveredVar(v.symbol)}
              onMouseLeave={() => setHoveredVar(null)}
              onClick={() => onVariableClick?.(v.symbol)}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-sm text-xs font-medium transition-all cursor-pointer border ${
                highlightedVar === v.symbol
                  ? 'border-blue-300 bg-blue-50 scale-105'
                  : 'border-gray-300 bg-gray-100 hover:bg-gray-200'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: v.color }}
              />
              <span style={{ color: v.color }} className="font-mono font-bold">
                {v.symbol}
              </span>
              <span className="text-gray-600">= {v.label}</span>
            </button>
          ))}
        </div>
      )}

      {hoveredVar && (
        <div className="text-xs text-gray-600 bg-gray-100 rounded-sm px-3 py-2 border border-gray-300">
          {variables.find((v) => v.symbol === hoveredVar)?.description}
        </div>
      )}
    </div>
  )
}
