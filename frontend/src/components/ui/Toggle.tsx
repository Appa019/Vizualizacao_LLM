interface ToggleProps {
  enabled: boolean
  onChange: (enabled: boolean) => void
  labelLeft?: string
  labelRight?: string
  size?: 'sm' | 'md'
}

export default function Toggle({
  enabled,
  onChange,
  labelLeft,
  labelRight,
  size = 'md',
}: ToggleProps) {
  const sizes = {
    sm: { track: 'w-9 h-5', thumb: 'w-4 h-4', translate: 'translate-x-4' },
    md: { track: 'w-11 h-6', thumb: 'w-5 h-5', translate: 'translate-x-5' },
  }
  const s = sizes[size]

  return (
    <div className="flex items-center gap-2">
      {labelLeft && (
        <span className={`text-xs font-medium ${!enabled ? 'text-gray-900' : 'text-gray-500'}`}>
          {labelLeft}
        </span>
      )}
      <button
        onClick={() => onChange(!enabled)}
        className={`relative inline-flex ${s.track} items-center rounded-full transition-colors duration-200 ${
          enabled ? 'bg-blue-600' : 'bg-gray-300'
        }`}
      >
        <span
          className={`inline-block ${s.thumb} transform rounded-full bg-white transition-transform duration-200 ${
            enabled ? s.translate : 'translate-x-0.5'
          }`}
        />
      </button>
      {labelRight && (
        <span className={`text-xs font-medium ${enabled ? 'text-gray-900' : 'text-gray-500'}`}>
          {labelRight}
        </span>
      )}
    </div>
  )
}
