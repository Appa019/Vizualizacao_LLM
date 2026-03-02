import { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

interface LossSurfaceProps {
  surfaceData?: number[][]
  trajectoryPoints?: [number, number, number][]
  height?: number
  gridSize?: number
  animateTrajectory?: boolean
}

function Surface({ data, gridSize }: { data: number[][]; gridSize: number }) {
  const geometry = useMemo(() => {
    const geom = new THREE.PlaneGeometry(6, 6, gridSize - 1, gridSize - 1)
    const positions = geom.attributes.position
    const colors = new Float32Array(positions.count * 3)

    let maxZ = -Infinity
    let minZ = Infinity
    for (let i = 0; i < positions.count; i++) {
      const ix = i % gridSize
      const iy = Math.floor(i / gridSize)
      const z = data[iy]?.[ix] ?? 0
      positions.setZ(i, z)
      maxZ = Math.max(maxZ, z)
      minZ = Math.min(minZ, z)
    }

    for (let i = 0; i < positions.count; i++) {
      const z = positions.getZ(i)
      const t = (z - minZ) / (maxZ - minZ || 1)
      // Blue (low) -> Yellow -> Red (high)
      const color = new THREE.Color()
      if (t < 0.5) {
        color.setRGB(t * 2, t * 2, 1 - t * 2)
      } else {
        color.setRGB(1, 1 - (t - 0.5) * 2, 0)
      }
      colors[i * 3] = color.r
      colors[i * 3 + 1] = color.g
      colors[i * 3 + 2] = color.b
    }

    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geom.computeVertexNormals()
    return geom
  }, [data, gridSize])

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
      <meshStandardMaterial vertexColors transparent opacity={0.8} side={THREE.DoubleSide} />
    </mesh>
  )
}

function TrajectoryBall({ points, animate }: { points: [number, number, number][]; animate: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [progress, setProgress] = useState(0)

  useFrame((_, delta) => {
    if (!animate || points.length < 2) return
    setProgress((prev) => {
      const next = prev + delta * 0.3
      return next > points.length - 1 ? 0 : next
    })

    if (meshRef.current) {
      const idx = Math.floor(progress)
      const frac = progress - idx
      const p1 = points[idx]
      const p2 = points[Math.min(idx + 1, points.length - 1)]
      meshRef.current.position.set(
        p1[0] + (p2[0] - p1[0]) * frac,
        p1[1] + (p2[1] - p1[1]) * frac - 0.85,
        p1[2] + (p2[2] - p1[2]) * frac
      )
    }
  })

  if (points.length < 2) return null

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
    </mesh>
  )
}

function Scene({ data, gridSize, trajectoryPoints, animateTrajectory }: {
  data: number[][]
  gridSize: number
  trajectoryPoints: [number, number, number][]
  animateTrajectory: boolean
}) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[5, 10, 5]} intensity={0.8} />
      <Surface data={data} gridSize={gridSize} />
      <TrajectoryBall points={trajectoryPoints} animate={animateTrajectory} />
      <gridHelper args={[6, 12, '#e5e7eb', '#f3f4f6']} position={[0, -1.5, 0]} />
      <OrbitControls enableDamping dampingFactor={0.05} minDistance={3} maxDistance={15} />
    </>
  )
}

function generateDefaultSurface(size: number): number[][] {
  const data: number[][] = []
  for (let i = 0; i < size; i++) {
    const row: number[] = []
    for (let j = 0; j < size; j++) {
      const x = (i / size - 0.5) * 6
      const y = (j / size - 0.5) * 6
      // Multi-modal loss surface
      const z =
        Math.sin(x) * Math.cos(y) * 0.5 +
        Math.exp(-((x - 1) ** 2 + (y - 1) ** 2) * 0.5) * 2 +
        Math.exp(-((x + 1.5) ** 2 + (y + 0.5) ** 2) * 0.3) * 1.5 +
        0.5
      row.push(z)
    }
    data.push(row)
  }
  return data
}

function generateDefaultTrajectory(): [number, number, number][] {
  const points: [number, number, number][] = []
  let x = 2.5
  let z = 2.5
  for (let i = 0; i < 50; i++) {
    const gradX = Math.cos(x) * Math.cos(z) * 0.1
    const gradZ = -Math.sin(x) * Math.sin(z) * 0.1
    x -= gradX * 0.8
    z -= gradZ * 0.8
    const y =
      Math.sin(x) * Math.cos(z) * 0.5 +
      Math.exp(-((x - 1) ** 2 + (z - 1) ** 2) * 0.5) * 2 +
      0.5
    points.push([x - 3, y, z - 3])
  }
  return points
}

export default function LossSurface({
  surfaceData,
  trajectoryPoints,
  height = 500,
  gridSize = 40,
  animateTrajectory = true,
}: LossSurfaceProps) {
  const data = useMemo(() => surfaceData || generateDefaultSurface(gridSize), [surfaceData, gridSize])
  const trajectory = useMemo(() => trajectoryPoints || generateDefaultTrajectory(), [trajectoryPoints])

  return (
    <div className="space-y-2">
      <div className="rounded-sm overflow-hidden border border-gray-200" style={{ height }}>
        <Canvas camera={{ position: [5, 4, 5], fov: 50 }}>
          <Scene
            data={data}
            gridSize={gridSize}
            trajectoryPoints={trajectory}
            animateTrajectory={animateTrajectory}
          />
        </Canvas>
      </div>
      <p className="text-xs text-gray-500 text-center">
        A bolinha vermelha representa o gradient descent descendo a superficie de loss.
      </p>
    </div>
  )
}
