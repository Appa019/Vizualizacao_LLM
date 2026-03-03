import { useRef, useState, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Html, Line } from '@react-three/drei'
import * as THREE from 'three'

interface EmbeddingPoint {
  word: string
  position: [number, number, number]
  category: string
}

interface EmbeddingSpaceProps {
  points: EmbeddingPoint[]
  height?: number
}

const CATEGORY_COLORS: Record<string, string> = {
  animal: '#ef4444',
  cor: '#3b82f6',
  verbo: '#22c55e',
  substantivo: '#f59e0b',
  adjetivo: '#a855f7',
  preposicao: '#6b7280',
  artigo: '#64748b',
  default: '#8b5cf6',
}

function Point({ point, isHovered, onHover }: {
  point: EmbeddingPoint
  isHovered: boolean
  onHover: (word: string | null) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const color = CATEGORY_COLORS[point.category] || CATEGORY_COLORS.default

  useFrame((_, delta) => {
    if (meshRef.current) {
      if (isHovered) {
        meshRef.current.scale.lerp(new THREE.Vector3(1.5, 1.5, 1.5), delta * 5)
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), delta * 5)
      }
    }
  })

  return (
    <group position={point.position}>
      <mesh
        ref={meshRef}
        onPointerOver={() => onHover(point.word)}
        onPointerOut={() => onHover(null)}
      >
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isHovered ? 0.5 : 0.2}
          transparent
          opacity={isHovered ? 1 : 0.8}
        />
      </mesh>
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.14}
        color={isHovered ? '#111827' : '#6b7280'}
        anchorX="center"
        anchorY="bottom"
        outlineWidth={0.015}
        outlineColor="#ffffff"
      >
        {point.word}
      </Text>
      {isHovered && (
        <Html center distanceFactor={8}>
          <div className="bg-white/90 border border-gray-200 rounded-sm px-3 py-2 text-xs text-gray-900 whitespace-nowrap backdrop-blur-sm shadow-lg">
            <div className="font-semibold">{point.word}</div>
            <div className="text-gray-600">Categoria: {point.category}</div>
            <div className="text-gray-600 font-mono">
              [{point.position.map((p) => p.toFixed(2)).join(', ')}]
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}

function CategoryLines({ points }: { points: EmbeddingPoint[] }) {
  const lines = useMemo(() => {
    const byCategory: Record<string, [number, number, number][]> = {}
    for (const p of points) {
      if (!byCategory[p.category]) byCategory[p.category] = []
      byCategory[p.category].push(p.position)
    }
    const result: { color: string; positions: [number, number, number][] }[] = []
    for (const [cat, positions] of Object.entries(byCategory)) {
      if (positions.length < 2) continue
      const color = CATEGORY_COLORS[cat] || CATEGORY_COLORS.default
      // Connect each point to the next in the group
      for (let i = 0; i < positions.length - 1; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          result.push({ color, positions: [positions[i], positions[j]] })
        }
      }
    }
    return result
  }, [points])

  return (
    <>
      {lines.map((line, i) => (
        <Line
          key={i}
          points={line.positions}
          color={line.color}
          lineWidth={1}
          dashed
          dashScale={8}
          dashSize={1}
          gapSize={1}
          transparent
          opacity={0.35}
        />
      ))}
    </>
  )
}

function Scene({ points }: { points: EmbeddingPoint[] }) {
  const [hoveredWord, setHoveredWord] = useState<string | null>(null)

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />

      {/* Axes */}
      <group>
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([-3, 0, 0, 3, 0, 0])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#d1d5db" />
        </line>
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([0, -3, 0, 0, 3, 0])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#d1d5db" />
        </line>
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([0, 0, -3, 0, 0, 3])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#d1d5db" />
        </line>
      </group>

      {/* Grid plane */}
      <gridHelper args={[6, 12, '#e5e7eb', '#f3f4f6']} rotation={[0, 0, 0]} />

      {/* Dashed lines connecting same-category words */}
      <CategoryLines points={points} />

      {points.map((point) => (
        <Point
          key={point.word}
          point={point}
          isHovered={hoveredWord === point.word}
          onHover={setHoveredWord}
        />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={2}
        maxDistance={15}
        autoRotate
        autoRotateSpeed={0.5}
      />
    </>
  )
}

export default function EmbeddingSpace({ points, height = 500 }: EmbeddingSpaceProps) {
  const categories = useMemo(() => {
    const cats = new Set(points.map((p) => p.category))
    return Array.from(cats)
  }, [points])

  return (
    <div className="space-y-3">
      <div className="rounded-sm overflow-hidden border border-gray-200" style={{ height }}>
        <Canvas camera={{ position: [4, 3, 4], fov: 50 }}>
          <Scene points={points} />
        </Canvas>
      </div>

      {/* Legend panel */}
      <div className="border border-gray-200 rounded-sm bg-white p-3">
        <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mb-2">Categorias</p>
        <div className="flex flex-wrap gap-x-5 gap-y-2">
          {categories.map((cat) => {
            const color = CATEGORY_COLORS[cat] || CATEGORY_COLORS.default
            const count = points.filter((p) => p.category === cat).length
            return (
              <div key={cat} className="flex items-center gap-2 text-xs text-gray-700">
                <span
                  className="w-3 h-3 rounded-full border"
                  style={{ backgroundColor: color, borderColor: color }}
                />
                <span className="font-medium">{cat}</span>
                <span className="text-gray-400 font-mono text-[10px]">({count})</span>
              </div>
            )
          })}
        </div>
        <p className="text-[10px] text-gray-400 mt-2">
          Linhas tracejadas conectam palavras da mesma categoria. Arraste para rotacionar. Scroll para zoom.
        </p>
      </div>
    </div>
  )
}
