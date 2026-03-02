import { useRef, useState, useEffect, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

interface InferencePipeline3DProps {
  tokens: string[]
  probabilities?: number[][]
  autoPlay?: boolean
  height?: number
}

const TOKEN_COLORS = ['#3b82f6', '#8b5cf6', '#6366f1', '#a855f7', '#2563eb', '#7c3aed']

function TokenBlock({
  token,
  index,
  animate,
}: {
  token: string
  index: number
  animate: boolean
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const scaleRef = useRef(animate ? 0 : 1)

  useFrame((_, delta) => {
    if (!meshRef.current) return
    if (scaleRef.current < 1) {
      scaleRef.current = Math.min(1, scaleRef.current + delta * 3)
      const s = scaleRef.current
      meshRef.current.scale.set(s, s, s)
    }
  })

  const color = TOKEN_COLORS[index % TOKEN_COLORS.length]
  const x = index * 1.6

  return (
    <group position={[x, 0, 0]}>
      <mesh ref={meshRef} scale={animate ? [0, 0, 0] : [1, 1, 1]}>
        <boxGeometry args={[1.2, 0.6, 0.6]} />
        <meshStandardMaterial color={color} />
      </mesh>
      <Text
        position={[0, 0, 0.31]}
        fontSize={0.18}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.015}
        outlineColor="#ffffff"
        maxWidth={1.0}
      >
        {token.length > 10 ? token.slice(0, 8) + '..' : token}
      </Text>
      <Text
        position={[0, -0.5, 0]}
        fontSize={0.1}
        color="#9ca3af"
        anchorX="center"
        anchorY="top"
      >
        {`[${index}]`}
      </Text>
    </group>
  )
}

function CandidateSpheres({
  candidates,
  xPosition,
}: {
  candidates: number[]
  xPosition: number
}) {
  const groupRef = useRef<THREE.Group>(null)

  useFrame((state) => {
    if (!groupRef.current) return
    groupRef.current.position.y =
      1.2 + Math.sin(state.clock.elapsedTime * 2) * 0.1
  })

  const sorted = candidates
    .map((prob, i) => ({ prob, index: i }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5)

  return (
    <group ref={groupRef} position={[xPosition, 1.2, 0]}>
      {sorted.map((candidate, i) => {
        const offsetX = (i - 2) * 0.5
        const offsetY = i * 0.35
        const radius = 0.1 + candidate.prob * 0.3
        const opacity = 0.2 + candidate.prob * 0.6

        return (
          <group key={i} position={[offsetX, offsetY, 0]}>
            <mesh>
              <sphereGeometry args={[radius, 16, 16]} />
              <meshStandardMaterial
                color={TOKEN_COLORS[candidate.index % TOKEN_COLORS.length]}
                transparent
                opacity={opacity}
                emissive={TOKEN_COLORS[candidate.index % TOKEN_COLORS.length]}
                emissiveIntensity={0.3}
              />
            </mesh>
            <Text
              position={[radius + 0.08, 0, 0]}
              fontSize={0.09}
              color="#6b7280"
              anchorX="left"
              anchorY="middle"
            >
              {`${(candidate.prob * 100).toFixed(0)}%`}
            </Text>
          </group>
        )
      })}
    </group>
  )
}

function CameraFollower({ targetX }: { targetX: number }) {
  const { camera } = useThree()
  const currentX = useRef(0)

  useFrame((_, delta) => {
    currentX.current += (targetX - currentX.current) * delta * 2
    camera.position.x = currentX.current
    camera.lookAt(currentX.current, 0, 0)
  })

  return null
}

function Scene({
  tokens,
  probabilities,
  visibleCount,
}: {
  tokens: string[]
  probabilities?: number[][]
  visibleCount: number
}) {
  const showCandidates =
    probabilities &&
    visibleCount < tokens.length &&
    visibleCount > 0 &&
    probabilities[visibleCount]

  const cameraTargetX = Math.max(0, (visibleCount - 1) * 1.6)

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, 5, -10]} intensity={0.3} />

      <gridHelper
        args={[Math.max(20, tokens.length * 2), 20, '#e5e7eb', '#f3f4f6']}
        position={[tokens.length * 0.8, -0.8, 0]}
      />

      {tokens.slice(0, visibleCount).map((token, i) => (
        <TokenBlock
          key={`${token}-${i}`}
          token={token}
          index={i}
          animate={i === visibleCount - 1}
        />
      ))}

      {showCandidates && (
        <CandidateSpheres
          candidates={probabilities[visibleCount]}
          xPosition={visibleCount * 1.6}
        />
      )}

      <CameraFollower targetX={cameraTargetX} />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={2}
        maxDistance={20}
        target={[cameraTargetX, 0, 0]}
      />
    </>
  )
}

export default function InferencePipeline3D({
  tokens,
  probabilities,
  autoPlay = true,
  height = 400,
}: InferencePipeline3DProps) {
  const [visibleCount, setVisibleCount] = useState(autoPlay ? 0 : tokens.length)
  const [isPlaying, setIsPlaying] = useState(autoPlay)

  useEffect(() => {
    if (!isPlaying || visibleCount >= tokens.length) {
      if (visibleCount >= tokens.length) setIsPlaying(false)
      return
    }
    const timer = setTimeout(() => setVisibleCount((c) => c + 1), 800)
    return () => clearTimeout(timer)
  }, [isPlaying, visibleCount, tokens.length])

  useEffect(() => {
    setVisibleCount(autoPlay ? 0 : tokens.length)
    setIsPlaying(autoPlay)
  }, [tokens, autoPlay])

  const togglePlay = useCallback(() => {
    if (visibleCount >= tokens.length) {
      setVisibleCount(0)
      setIsPlaying(true)
    } else {
      setIsPlaying((p) => !p)
    }
  }, [visibleCount, tokens.length])

  const reset = useCallback(() => {
    setVisibleCount(0)
    setIsPlaying(true)
  }, [])

  return (
    <div className="space-y-3">
      <div
        className="rounded-sm overflow-hidden border border-gray-200"
        style={{ height }}
      >
        <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
          <Scene
            tokens={tokens}
            probabilities={probabilities}
            visibleCount={visibleCount}
          />
        </Canvas>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={togglePlay}
          className="px-3 py-1.5 rounded-sm bg-blue-600 hover:bg-blue-500 text-white text-xs transition-colors"
        >
          {isPlaying ? 'Pausar' : visibleCount >= tokens.length ? 'Reproduzir' : 'Continuar'}
        </button>
        <button
          onClick={reset}
          className="px-3 py-1.5 rounded-sm bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs transition-colors"
        >
          Reiniciar
        </button>
        <span className="text-xs text-gray-500 ml-2">
          {visibleCount} / {tokens.length} tokens
        </span>
      </div>

      <p className="text-xs text-gray-500">
        Observe como o modelo gera texto token a token.
      </p>
    </div>
  )
}
