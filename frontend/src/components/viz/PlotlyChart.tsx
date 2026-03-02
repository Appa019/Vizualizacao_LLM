import { lazy, Suspense } from 'react'

const Plot = lazy(() => import('react-plotly.js'))

interface PlotlyChartProps {
  data: Plotly.Data[]
  layout?: Partial<Plotly.Layout>
  config?: Partial<Plotly.Config>
  className?: string
  height?: number
}

const chartLayout: Partial<Plotly.Layout> = {
  paper_bgcolor: 'rgba(255,255,255,0)',
  plot_bgcolor: 'rgba(255,255,255,0)',
  font: { color: '#374151', family: 'Inter, system-ui, sans-serif', size: 11 },
  xaxis: {
    gridcolor: 'rgba(209,213,219,0.8)',
    zerolinecolor: 'rgba(209,213,219,1)',
  },
  yaxis: {
    gridcolor: 'rgba(209,213,219,0.8)',
    zerolinecolor: 'rgba(209,213,219,1)',
  },
  margin: { t: 50, r: 40, b: 60, l: 60 },
  hoverlabel: {
    bgcolor: '#ffffff',
    bordercolor: '#e5e7eb',
    font: { color: '#111827', family: 'Inter, system-ui, sans-serif' },
  },
}

const defaultConfig: Partial<Plotly.Config> = {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d'],
  responsive: true,
}

export default function PlotlyChart({
  data,
  layout = {},
  config = {},
  className = '',
  height = 400,
}: PlotlyChartProps) {
  const mergedLayout: Partial<Plotly.Layout> = {
    ...chartLayout,
    ...layout,
    height,
    xaxis: { ...chartLayout.xaxis, ...(layout.xaxis || {}) },
    yaxis: { ...chartLayout.yaxis, ...(layout.yaxis || {}) },
    font: { ...chartLayout.font, ...(layout.font || {}) },
    margin: { ...chartLayout.margin, ...(layout.margin || {}) },
  }

  const mergedConfig = { ...defaultConfig, ...config }

  return (
    <div className={`w-full ${className}`}>
      <Suspense
        fallback={
          <div
            className="flex items-center justify-center bg-gray-100 rounded-sm animate-pulse"
            style={{ height }}
          >
            <span className="text-gray-500 text-sm">Carregando grafico...</span>
          </div>
        }
      >
        <Plot
          data={data}
          layout={mergedLayout}
          config={mergedConfig}
          useResizeHandler
          className="w-full"
          style={{ width: '100%' }}
        />
      </Suspense>
    </div>
  )
}
