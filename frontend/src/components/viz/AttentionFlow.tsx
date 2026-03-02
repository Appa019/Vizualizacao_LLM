import { useMemo } from 'react'
import { motion } from 'framer-motion'

interface AttentionFlowProps {
  tokens: string[]
  weights: number[]
  sourceToken: number
  maxTokens?: number
}

export default function AttentionFlow({
  tokens,
  weights,
  sourceToken,
  maxTokens,
}: AttentionFlowProps) {
  const displayTokens = maxTokens ? tokens.slice(0, maxTokens) : tokens
  const displayWeights = maxTokens ? weights.slice(0, maxTokens) : weights

  const maxWeight = Math.max(...displayWeights)
  const fontSize = displayTokens.length > 6 ? 9 : 11
  const tokenSpacing = 800 / Math.max(displayTokens.length, 1)

  const lines = useMemo(() => {
    const sourceX = 50 + sourceToken * tokenSpacing
    const sourceY = 60

    return displayWeights.map((w, i) => {
      if (i === sourceToken || w < 0.02) return null
      const targetX = 50 + i * tokenSpacing
      const targetY = 200
      const opacity = Math.min(1, w / maxWeight)
      const strokeWidth = 1 + (w / maxWeight) * 4

      return { sourceX, sourceY, targetX, targetY, opacity, strokeWidth, weight: w, index: i }
    }).filter(Boolean)
  }, [displayWeights, sourceToken, tokenSpacing, maxWeight])

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox="0 0 900 280" className="w-full min-w-[600px]">
        {/* Source token row */}
        <g>
          {displayTokens.map((token, i) => {
            const x = 50 + i * tokenSpacing
            const isSource = i === sourceToken
            return (
              <g key={`top-${i}`}>
                <rect
                  x={x - 30}
                  y={30}
                  width={60}
                  height={30}
                  rx={6}
                  fill={isSource ? '#3b82f6' : '#f3f4f6'}
                  stroke={isSource ? '#60a5fa' : '#e5e7eb'}
                  strokeWidth={isSource ? 2 : 1}
                />
                <text
                  x={x}
                  y={50}
                  textAnchor="middle"
                  fill={isSource ? '#ffffff' : '#6b7280'}
                  fontSize={fontSize}
                  fontFamily="Inter, sans-serif"
                >
                  {token}
                </text>
              </g>
            )
          })}
        </g>

        {/* Connection lines */}
        {lines.map((line) => {
          if (!line) return null
          return (
            <motion.path
              key={`line-${line.index}`}
              d={`M ${line.sourceX} ${line.sourceY} C ${line.sourceX} 130, ${line.targetX} 130, ${line.targetX} ${line.targetY}`}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={line.strokeWidth}
              opacity={line.opacity}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.8, delay: line.index * 0.05 }}
            />
          )
        })}

        {/* Target token row with bar chart */}
        <g>
          {displayTokens.map((token, i) => {
            const x = 50 + i * tokenSpacing
            const barHeight = (displayWeights[i] / maxWeight) * 40
            const isSource = i === sourceToken

            return (
              <g key={`bottom-${i}`}>
                {/* Bar */}
                <motion.rect
                  x={x - 15}
                  y={200 - barHeight}
                  width={30}
                  height={barHeight}
                  rx={3}
                  fill={isSource ? '#ef4444' : '#3b82f6'}
                  opacity={0.7}
                  initial={{ height: 0, y: 200 }}
                  animate={{ height: barHeight, y: 200 - barHeight }}
                  transition={{ duration: 0.5, delay: i * 0.05 }}
                />
                {/* Weight value */}
                <text
                  x={x}
                  y={195 - barHeight}
                  textAnchor="middle"
                  fill="#6b7280"
                  fontSize={9}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {displayWeights[i].toFixed(2)}
                </text>
                {/* Token label */}
                <rect
                  x={x - 30}
                  y={210}
                  width={60}
                  height={30}
                  rx={6}
                  fill={isSource ? '#fef2f2' : '#f3f4f6'}
                  stroke={isSource ? '#ef4444' : '#e5e7eb'}
                  strokeWidth={1}
                />
                <text
                  x={x}
                  y={230}
                  textAnchor="middle"
                  fill={isSource ? '#b91c1c' : '#6b7280'}
                  fontSize={fontSize}
                  fontFamily="Inter, sans-serif"
                >
                  {token}
                </text>
              </g>
            )
          })}
        </g>

        {/* Labels */}
        <text x={10} y={50} fill="#6b7280" fontSize={10}>
          Query
        </text>
        <text x={10} y={230} fill="#6b7280" fontSize={10}>
          Keys
        </text>
      </svg>
    </div>
  )
}
