import { useState } from 'react'
import { Lightbulb, CaretDown, CaretUp } from '@phosphor-icons/react'

interface WhyItMattersProps {
  title?: string
  children: React.ReactNode
  defaultOpen?: boolean
}

export default function WhyItMatters({
  title = 'Por que isso importa?',
  children,
  defaultOpen = true,
}: WhyItMattersProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-amber-200 bg-amber-50 rounded-sm overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-amber-100 transition-colors"
      >
        <Lightbulb weight="fill" className="w-4 h-4 text-amber-500 flex-shrink-0" />
        <span className="text-sm font-semibold text-amber-800 flex-1">{title}</span>
        {isOpen ? (
          <CaretUp weight="bold" className="w-4 h-4 text-amber-500" />
        ) : (
          <CaretDown weight="bold" className="w-4 h-4 text-amber-500" />
        )}
      </button>
      {isOpen && (
        <div className="px-4 pb-4 text-sm text-amber-700 leading-relaxed">
          {children}
        </div>
      )}
    </div>
  )
}
