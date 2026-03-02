import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text, Html } from '@react-three/drei'
import { useState } from 'react'
import * as THREE from 'three'

interface Layer {
  name: string
  type: 'input' | 'embedding' | 'attention' | 'ffn' | 'norm' | 'output'
  description: string
  color: string
}

interface NetworkGraphProps {
  layers?: Layer[]
  height?: number
  onLayerClick?: (layer: Layer) => void
}

const DEFAULT_LAYERS: Layer[] = [
  { name: 'Input', type: 'input', description: 'Tokens de entrada', color: '#6b7280' },
  { name: 'Embedding', type: 'embedding', description: 'Token + Positional Embeddings', color: '#3b82f6' },
  { name: 'Layer Norm', type: 'norm', description: 'Normalizacao de camada', color: '#8b5cf6' },
  { name: 'Multi-Head\nAttention', type: 'attention', description: 'Self-Attention com multiplas cabecas', color: '#ef4444' },
  { name: 'Add & Norm', type: 'norm', description: 'Conexao residual + normalizacao', color: '#8b5cf6' },
  { name: 'Feed\nForward', type: 'ffn', description: 'Rede feed-forward (expandir + comprimir)', color: '#22c55e' },
  { name: 'Add & Norm', type: 'norm', description: 'Conexao residual + normalizacao', color: '#8b5cf6' },
  { name: 'Output', type: 'output', description: 'Logits / Probabilidades', color: '#f59e0b' },
]

function LayerBlock({ layer, position, onClick, isHovered, onHover }: {
  layer: Layer
  position: [number, number, number]
  onClick?: () => void
  isHovered: boolean
  onHover: (name: string | null) => void
}) {
  return (
    <group position={position}>
      <mesh
        onClick={onClick}
        onPointerOver={() => onHover(layer.name)}
        onPointerOut={() => onHover(null)}
      >
        <boxGeometry args={[2.5, 0.6, 1.5]} />
        <meshStandardMaterial
          color={layer.color}
          transparent
          opacity={isHovered ? 0.9 : 0.6}
          emissive={layer.color}
          emissiveIntensity={isHovered ? 0.3 : 0.1}
        />
      </mesh>
      <Text
        position={[0, 0, 0.8]}
        fontSize={0.18}
        color="#111827"
        anchorX="center"
        anchorY="middle"
        maxWidth={2}
        outlineWidth={0.015}
        outlineColor="#ffffff"
      >
        {layer.name}
      </Text>
      {isHovered && (
        <Html center distanceFactor={6} position={[2, 0, 0]}>
          <div className="bg-white/95 border border-gray-200 rounded-sm px-3 py-2 text-xs text-gray-900 whitespace-nowrap backdrop-blur-sm shadow-lg min-w-[150px]">
            <div className="font-semibold" style={{ color: layer.color }}>{layer.name}</div>
            <div className="text-gray-600 mt-1">{layer.description}</div>
          </div>
        </Html>
      )}
    </group>
  )
}

function Arrow({ from, to }: { from: [number, number, number]; to: [number, number, number] }) {
  const midY = (from[1] + to[1]) / 2
  const points = [
    new THREE.Vector3(...from),
    new THREE.Vector3(from[0], midY, from[2]),
    new THREE.Vector3(to[0], midY, to[2]),
    new THREE.Vector3(...to),
  ]
  const curve = new THREE.CatmullRomCurve3(points)
  const geometry = new THREE.TubeGeometry(curve, 20, 0.03, 8, false)

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color="#4b5563" emissive="#4b5563" emissiveIntensity={0.2} />
    </mesh>
  )
}

function Scene({ layers, onLayerClick }: { layers: Layer[]; onLayerClick?: (l: Layer) => void }) {
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null)
  const spacing = 1.2

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[5, 10, 5]} intensity={0.8} />
      <pointLight position={[-5, -5, 5]} intensity={0.3} />

      {layers.map((layer, i) => {
        const y = (layers.length / 2 - i) * spacing
        return (
          <LayerBlock
            key={`${layer.name}-${i}`}
            layer={layer}
            position={[0, y, 0]}
            onClick={() => onLayerClick?.(layer)}
            isHovered={hoveredLayer === layer.name}
            onHover={setHoveredLayer}
          />
        )
      })}

      {/* Arrows between layers */}
      {layers.slice(0, -1).map((_, i) => {
        const y1 = (layers.length / 2 - i) * spacing - 0.35
        const y2 = (layers.length / 2 - (i + 1)) * spacing + 0.35
        return (
          <Arrow
            key={`arrow-${i}`}
            from={[0, y1, 0]}
            to={[0, y2, 0]}
          />
        )
      })}

      {/* Residual connections */}
      {[2, 5].map((i) => {
        if (i + 2 >= layers.length) return null
        const y1 = (layers.length / 2 - i) * spacing
        const y2 = (layers.length / 2 - (i + 2)) * spacing
        return (
          <Arrow
            key={`res-${i}`}
            from={[-1.5, y1, 0]}
            to={[-1.5, y2, 0]}
          />
        )
      })}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={4}
        maxDistance={20}
      />
    </>
  )
}

export default function NetworkGraph({
  layers = DEFAULT_LAYERS,
  height = 600,
  onLayerClick,
}: NetworkGraphProps) {
  return (
    <div className="space-y-2">
      <div className="rounded-sm overflow-hidden border border-gray-200" style={{ height }}>
        <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
          <Scene layers={layers} onLayerClick={onLayerClick} />
        </Canvas>
      </div>
      <p className="text-xs text-gray-500 text-center">
        Arraste para rotacionar. Clique em uma camada para ver detalhes.
      </p>
    </div>
  )
}
