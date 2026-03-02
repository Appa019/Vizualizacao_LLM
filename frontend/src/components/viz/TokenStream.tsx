import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface TokenStreamProps {
  tokens: string[]
  probabilities?: number[][]
  speed?: number
  autoPlay?: boolean
}

const TOKEN_COLORS = [
  'bg-blue-50 border-blue-200 text-blue-700',
  'bg-purple-50 border-purple-200 text-purple-700',
  'bg-green-50 border-green-200 text-green-700',
  'bg-amber-50 border-amber-200 text-amber-700',
  'bg-red-50 border-red-200 text-red-700',
  'bg-cyan-50 border-cyan-200 text-cyan-700',
  'bg-pink-50 border-pink-200 text-pink-700',
  'bg-indigo-50 border-indigo-200 text-indigo-700',
]

export default function TokenStream({
  tokens,
  probabilities,
  speed = 500,
  autoPlay = true,
}: TokenStreamProps) {
  const [visibleCount, setVisibleCount] = useState(autoPlay ? 0 : tokens.length)
  const [isPlaying, setIsPlaying] = useState(autoPlay)

  useEffect(() => {
    if (!isPlaying || visibleCount >= tokens.length) {
      if (visibleCount >= tokens.length) setIsPlaying(false)
      return
    }
    const timer = setTimeout(() => setVisibleCount((c) => c + 1), speed)
    return () => clearTimeout(timer)
  }, [isPlaying, visibleCount, tokens.length, speed])

  const reset = () => {
    setVisibleCount(0)
    setIsPlaying(true)
  }

  return (
    <div className="space-y-4">
      {/* Token display */}
      <div className="flex flex-wrap gap-2 min-h-[60px] p-4 glass-card">
        <AnimatePresence>
          {tokens.slice(0, visibleCount).map((token, i) => (
            <motion.span
              key={`${token}-${i}`}
              initial={{ opacity: 0, scale: 0.5, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              className={`inline-flex items-center px-3 py-1.5 rounded-sm border text-sm font-mono ${
                TOKEN_COLORS[i % TOKEN_COLORS.length]
              }`}
            >
              {token.length > 15 ? token.slice(0, 12) + '...' : token}
              {probabilities && probabilities[i] && (
                <span className="ml-2 text-[10px] opacity-60">
                  {(Math.max(...probabilities[i]) * 100).toFixed(0)}%
                </span>
              )}
            </motion.span>
          ))}
        </AnimatePresence>
        {visibleCount < tokens.length && isPlaying && (
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1, repeat: Infinity }}
            className="inline-flex items-center px-2 py-1.5 text-gray-500"
          >
            |
          </motion.span>
        )}
      </div>

      {/* Probability bars for latest token */}
      {probabilities && visibleCount > 0 && probabilities[visibleCount - 1] && (
        <div className="glass-card p-4 space-y-2">
          <div className="text-xs text-gray-600 font-medium">
            Probabilidades para posicao {visibleCount}:
          </div>
          {probabilities[visibleCount - 1]
            .map((p, i) => ({ p, token: tokens[i] || `token_${i}`, i }))
            .sort((a, b) => b.p - a.p)
            .slice(0, 5)
            .map(({ p, token, i }) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-16 text-gray-600 font-mono truncate">{token}</span>
                <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${p * 100}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
                <span className="w-12 text-right text-gray-500 font-mono">
                  {(p * 100).toFixed(1)}%
                </span>
              </div>
            ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-3 py-1.5 rounded-sm bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          {isPlaying ? 'Pausar' : 'Continuar'}
        </button>
        <button
          onClick={reset}
          className="px-3 py-1.5 rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
        >
          Reiniciar
        </button>
        <span className="flex items-center text-gray-500">
          {visibleCount} / {tokens.length} tokens
        </span>
      </div>
    </div>
  )
}
