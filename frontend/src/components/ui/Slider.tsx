interface SliderProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  label?: string
  showValue?: boolean
  unit?: string
}

export default function Slider({
  value,
  onChange,
  min,
  max,
  step = 1,
  label,
  showValue = true,
  unit = '',
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className="space-y-1.5">
      {(label || showValue) && (
        <div className="flex items-center justify-between text-sm">
          {label && <span className="text-gray-600">{label}</span>}
          {showValue && (
            <span className="font-mono text-blue-400 font-medium">
              {value}
              {unit}
            </span>
          )}
        </div>
      )}
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`,
        }}
      />
      <div className="flex justify-between text-[10px] text-gray-400">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  )
}
