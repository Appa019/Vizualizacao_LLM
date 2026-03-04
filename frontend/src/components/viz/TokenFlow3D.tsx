import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import * as THREE from 'three'

interface TokenFlow3DProps {
  tokens?: string[]
  layers?: string[]
  speed?: number
  height?: number
}

const DEFAULT_TOKENS = ['o', 'gato', 'sentou', 'no']
const DEFAULT_LAYERS = ['Embedding', 'Self-Attention', 'Feed-Forward', 'Output']

const LAYER_COLORS: Record<string, string> = {
  Embedding: '#3b82f6',
  'Self-Attention': '#ef4444',
  'Feed-Forward': '#22c55e',
  Output: '#f59e0b',
}

function getLayerColor(layerName: string, layers: string[]): string {
  const defaultColors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b']
  if (LAYER_COLORS[layerName]) return LAYER_COLORS[layerName]
  const idx = layers.indexOf(layerName)
  return defaultColors[idx % defaultColors.length]
}

function LayerBox({
  position,
  label,
  color,
  width,
}: {
  position: [number, number, number]
  label: string
  color: string
  width: number
}) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={[width, 0.15, 2]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.15}
        />
      </mesh>
      {/* Border edges */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(width, 0.15, 2)]} />
        <lineBasicMaterial color={color} transparent opacity={0.4} />
      </lineSegments>
      <Text
        position={[width / 2 + 0.3, 0, 0]}
        fontSize={0.2}
        color={color}
        anchorX="left"
        anchorY="middle"

        outlineWidth={0.015}
        outlineColor="#ffffff"
      >
        {label}
      </Text>
    </group>
  )
}

function TokenSphere({
  token,
  index,
  totalTokens,
  layers,
  speed,
  layerSpacing,
  bottomY,
}: {
  token: string
  index: number
  totalTokens: number
  layers: string[]
  speed: number
  layerSpacing: number
  bottomY: number
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const colorRef = useRef(new THREE.Color(getLayerColor(layers[0], layers)))
  const progressRef = useRef(index / totalTokens)

  const totalHeight = (layers.length - 1) * layerSpacing
  const xOffset = (index - (totalTokens - 1) / 2) * 0.8

  useFrame((_, delta) => {
    if (!meshRef.current) return

    progressRef.current += delta * speed * 0.15
    if (progressRef.current > 1) {
      progressRef.current -= 1
    }

    const y = bottomY + progressRef.current * totalHeight
    meshRef.current.position.y = y

    const layerIndex = Math.min(
      Math.floor(progressRef.current * layers.length),
      layers.length - 1
    )
    const targetColor = new THREE.Color(getLayerColor(layers[layerIndex], layers))
    colorRef.current.lerp(targetColor, delta * 4)

    const mat = meshRef.current.material as THREE.MeshStandardMaterial
    mat.color.copy(colorRef.current)
    mat.emissive.copy(colorRef.current)
  })

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[xOffset, bottomY, 0]}
      >
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial
          color={getLayerColor(layers[0], layers)}
          emissive={getLayerColor(layers[0], layers)}
          emissiveIntensity={0.3}
          transparent
          opacity={0.9}
        />
      </mesh>
      <TokenLabel
        token={token}
        xOffset={xOffset}
        index={index}
        totalTokens={totalTokens}
        layers={layers}
        speed={speed}
        layerSpacing={layerSpacing}
        bottomY={bottomY}
        totalHeight={totalHeight}
      />
    </group>
  )
}

function TokenLabel({
  token,
  xOffset,
  index,
  totalTokens,
  layers,
  speed,
  layerSpacing,
  bottomY,
  totalHeight,
}: {
  token: string
  xOffset: number
  index: number
  totalTokens: number
  layers: string[]
  speed: number
  layerSpacing: number
  bottomY: number
  totalHeight: number
}) {
  const groupRef = useRef<THREE.Group>(null)
  const progressRef = useRef(index / totalTokens)

  useFrame((_, delta) => {
    if (!groupRef.current) return

    progressRef.current += delta * speed * 0.15
    if (progressRef.current > 1) {
      progressRef.current -= 1
    }

    const y = bottomY + progressRef.current * totalHeight
    groupRef.current.position.y = y + 0.25
  })

  return (
    <group ref={groupRef} position={[xOffset, bottomY + 0.25, 0]}>
      <Text
        fontSize={0.12}
        color="#374151"
        anchorX="center"
        anchorY="bottom"

        outlineWidth={0.015}
        outlineColor="#ffffff"
      >
        {token}
      </Text>
    </group>
  )
}

function Scene({
  tokens,
  layers,
  speed,
}: {
  tokens: string[]
  layers: string[]
  speed: number
}) {
  const layerSpacing = 2.0
  const bottomY = -((layers.length - 1) * layerSpacing) / 2

  const layerPositions = useMemo(() => {
    return layers.map((_, i) => bottomY + i * layerSpacing)
  }, [layers, bottomY, layerSpacing])

  const boxWidth = Math.max(tokens.length * 0.8 + 1, 3)

  return (
    <>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} />

      {/* Layer boxes */}
      {layers.map((layer, i) => (
        <LayerBox
          key={layer}
          position={[0, layerPositions[i], 0]}
          label={layer}
          color={getLayerColor(layer, layers)}
          width={boxWidth}
        />
      ))}

      {/* Vertical guide lines */}
      {tokens.map((_, i) => {
        const xOffset = (i - (tokens.length - 1) / 2) * 0.8
        const points = new Float32Array([
          xOffset, bottomY - 0.5, 0,
          xOffset, bottomY + (layers.length - 1) * layerSpacing + 0.5, 0,
        ])
        return (
          <line key={`guide-${i}`}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={points}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#d1d5db" transparent opacity={0.3} />
          </line>
        )
      })}

      {/* Token spheres */}
      {tokens.map((token, i) => (
        <TokenSphere
          key={`${token}-${i}`}
          token={token}
          index={i}
          totalTokens={tokens.length}
          layers={layers}
          speed={speed}
          layerSpacing={layerSpacing}
          bottomY={bottomY}
        />
      ))}

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={15}
      />
    </>
  )
}

export default function TokenFlow3D({
  tokens = DEFAULT_TOKENS,
  layers = DEFAULT_LAYERS,
  speed = 1,
  height = 500,
}: TokenFlow3DProps) {
  return (
    <div className="space-y-3">
      <div className="rounded-sm overflow-hidden border border-gray-200" style={{ height }}>
        <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
          <Scene tokens={tokens} layers={layers} speed={speed} />
        </Canvas>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 justify-center">
        {layers.map((layer) => (
          <div key={layer} className="flex items-center gap-1.5 text-xs text-gray-600">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: getLayerColor(layer, layers) }}
            />
            {layer}
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500 text-center">
        Arraste para rotacionar. Os tokens fluem pelas camadas do Transformer.
      </p>
    </div>
  )
}
