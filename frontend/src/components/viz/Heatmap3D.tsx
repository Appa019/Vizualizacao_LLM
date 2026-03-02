import { useMemo } from 'react'
import PlotlyChart from './PlotlyChart'

interface Heatmap3DProps {
  matrix: number[][]
  xLabels: string[]
  yLabels: string[]
  title?: string
  colorscale?: string
  height?: number
  mode?: '2d' | '3d'
  showValues?: boolean
}

export default function Heatmap3D({
  matrix,
  xLabels,
  yLabels,
  title = '',
  colorscale = 'YlOrRd',
  height = 450,
  mode = '2d',
  showValues = true,
}: Heatmap3DProps) {
  const data = useMemo(() => {
    if (mode === '3d') {
      return [
        {
          type: 'surface' as const,
          z: matrix,
          x: xLabels,
          y: yLabels,
          colorscale,
          hovertemplate:
            'De: %{y}<br>Para: %{x}<br>Peso: %{z:.3f}<extra></extra>',
          contours: {
            z: { show: true, usecolormap: true, highlightcolor: '#ffffff', project: { z: true } },
          },
        },
      ] as unknown as Plotly.Data[]
    }
    return [
      {
        type: 'heatmap' as const,
        z: matrix,
        x: xLabels,
        y: yLabels,
        colorscale,
        hovertemplate:
          'De: %{y}<br>Para: %{x}<br>Peso: %{z:.3f}<extra></extra>',
        text: showValues
          ? matrix.map((row) => row.map((v) => v.toFixed(2)))
          : undefined,
        texttemplate: showValues ? '%{text}' : undefined,
        textfont: { size: 10, color: '#ffffff' },
      },
    ] as unknown as Plotly.Data[]
  }, [matrix, xLabels, yLabels, colorscale, mode, showValues])

  const layout = useMemo(() => {
    if (mode === '3d') {
      return {
        title: { text: title, font: { color: '#111827', size: 14 } },
        scene: {
          xaxis: { title: 'Keys', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          yaxis: { title: 'Queries', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          zaxis: { title: 'Peso', color: '#6b7280', gridcolor: 'rgba(209,213,219,0.8)' },
          bgcolor: 'rgba(255,255,255,0)',
        },
      } as Partial<Plotly.Layout>
    }
    return {
      title: { text: title, font: { color: '#111827', size: 14 } },
      xaxis: { title: 'Keys (Attending to)', side: 'bottom' as const, tickangle: -45 },
      yaxis: { title: 'Queries (Attending from)', autorange: 'reversed' as const },
      margin: { l: 80, r: 40, t: 50, b: 80 },
    } as Partial<Plotly.Layout>
  }, [title, mode])

  return <PlotlyChart data={data} layout={layout} height={height} />
}
