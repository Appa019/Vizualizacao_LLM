interface CardProps {
  children: React.ReactNode
  className?: string
  glow?: 'blue' | 'purple' | 'green' | 'amber' | 'none'
  hover?: boolean
  onClick?: () => void
}

export default function Card({
  children,
  className = '',
  glow = 'none',
  hover = false,
  onClick,
}: CardProps) {
  const glowClasses = {
    none: '',
    blue: 'glow-blue border-blue-500/20',
    purple: 'glow-purple border-purple-500/20',
    green: 'border-green-500/20 shadow-[0_0_20px_rgba(34,197,94,0.1)]',
    amber: 'border-amber-500/20 shadow-[0_0_20px_rgba(245,158,11,0.1)]',
  }

  return (
    <div
      onClick={onClick}
      className={`glass-card p-6 ${glowClasses[glow]} ${
        hover ? 'hover:bg-gray-50 hover:shadow-md hover:border-gray-300/60 cursor-pointer transition-all duration-300 hover:scale-[1.02]' : ''
      } ${onClick ? 'cursor-pointer' : ''} ${className}`}
    >
      {children}
    </div>
  )
}
