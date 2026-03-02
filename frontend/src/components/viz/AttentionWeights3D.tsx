import { useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, QuadraticBezierLine } from '@react-three/drei'
import * as THREE from 'three'
import { useRef } from 'react'

interface AttentionWeights3DProps {
  tokens: string[]
  weights: number[][]
  height?: number
}

interface TokenPosition {
  index: number
  token: string
  position: [number, number, number]
}

interface BeamData {
  from: number
  to: number
  start: [number, number, number]
  end: [number, number, number]
  mid: [number, number, number]
  weight: number
}

function TokenSphere({
  tokenPos,
  isHovered,
  isFaded,
  onHover,
  onUnhover,
}: {
  tokenPos: TokenPosition
  isHovered: boolean
  isFaded: boolean
  onHover: () => void
  onUnhover: () => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((_, delta) => {
    if (!meshRef.current) return
    const target = isHovered ? 1.4 : 1.0
    const current = meshRef.current.scale.x
    const next = THREE.MathUtils.lerp(current, target, delta * 6)
    meshRef.current.scale.setScalar(next)
  })

  const color = isHovered ? '#ef4444' : '#3b82f6'
  const opacity = isFaded ? 0.3 : 1.0

  return (
    <group position={tokenPos.position}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation()
          onHover()
        }}
        onPointerOut={(e) => {
          e.stopPropagation()
          onUnhover()
        }}
      >
        <sphereGeometry args={[0.2, 24, 24]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHovered ? 0.4 : 0.15}
          transparent
          opacity={opacity}
        />
      </mesh>
      <Text
        position={[0, 0.4, 0]}
        fontSize={0.22}
        color={isHovered ? '#111827' : '#374151'}
        anchorX="center"
        anchorY="bottom"
        font="/fonts/Inter-Medium.woff"
        outlineWidth={0.015}
        outlineColor="#ffffff"
      >
        {tokenPos.token}
      </Text>
    </group>
  )
}

function Scene({ tokens, weights }: { tokens: string[]; weights: number[][] }) {
  const [hoveredToken, setHoveredToken] = useState<number | null>(null)

  const radius = 2.5

  const tokenPositions: TokenPosition[] = useMemo(() => {
    return tokens.map((token, i) => {
      const angle = (i / tokens.length) * Math.PI * 2 - Math.PI / 2
      const x = Math.cos(angle) * radius
      const z = Math.sin(angle) * radius
      return { index: i, token, position: [x, 0, z] as [number, number, number] }
    })
  }, [tokens, radius])

  const beams: BeamData[] = useMemo(() => {
    const result: BeamData[] = []
    for (let from = 0; from < tokens.length; from++) {
      for (let to = 0; to < tokens.length; to++) {
        if (from === to) continue
        const w = weights[from]?.[to] ?? 0
        if (w <= 0.05) continue

        const start = tokenPositions[from].position
        const end = tokenPositions[to].position

        const midX = (start[0] + end[0]) / 2
        const midY = w * 2
        const midZ = (start[2] + end[2]) / 2

        result.push({
          from,
          to,
          start,
          end,
          mid: [midX, midY, midZ],
          weight: w,
        })
      }
    }
    return result
  }, [tokens.length, weights, tokenPositions])

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 8, 5]} intensity={0.7} />
      <pointLight position={[-5, 4, -5]} intensity={0.3} />

      <gridHelper args={[8, 16, '#e5e7eb', '#f3f4f6']} />

      {tokenPositions.map((tp) => {
        const isFaded = hoveredToken !== null && hoveredToken !== tp.index
        return (
          <TokenSphere
            key={tp.index}
            tokenPos={tp}
            isHovered={hoveredToken === tp.index}
            isFaded={isFaded}
            onHover={() => setHoveredToken(tp.index)}
            onUnhover={() => setHoveredToken(null)}
          />
        )
      })}

      {beams.map((beam) => {
        let opacity: number
        let lineWidth: number

        if (hoveredToken === null) {
          opacity = 0.2 + beam.weight * 0.6
          lineWidth = 1 + beam.weight * 3
        } else if (beam.from === hoveredToken) {
          opacity = 0.3 + beam.weight * 0.7
          lineWidth = 1.5 + beam.weight * 4
        } else {
          opacity = 0.1
          lineWidth = 0.5
        }

        const isActive = hoveredToken === null || beam.from === hoveredToken
        const color = isActive ? '#3b82f6' : '#9ca3af'

        return (
          <QuadraticBezierLine
            key={`${beam.from}-${beam.to}`}
            start={beam.start}
            end={beam.end}
            mid={beam.mid}
            color={color}
            lineWidth={lineWidth}
            transparent
            opacity={opacity}
/>
        )
      })}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={12}
      />
    </>
  )
}

export default function AttentionWeights3D({
  tokens,
  weights,
  height = 450,
}: AttentionWeights3DProps) {
  return (
    <div className="space-y-3">
      <div
        className="rounded-sm overflow-hidden border border-gray-200"
        style={{ height }}
      >
        <Canvas camera={{ position: [3, 4, 5], fov: 50 }}>
          <Scene tokens={tokens} weights={weights} />
        </Canvas>
      </div>
      <p className="text-xs text-gray-500 text-center">
        Passe o mouse sobre um token para ver seus pesos de atencao.
      </p>
    </div>
  )
}
