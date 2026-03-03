import { useState, useEffect, useCallback, useRef } from 'react'
import apiClient, {
  type TokenizationResponse,
  type BPEStepsResponse,
  type CompareTokenizersResponse,
  type EmbeddingResponse,
  type PositionalEncodingResponse,
  type EmbeddingSpaceResponse,
  type SelfAttentionResponse,
  type MultiHeadAttentionResponse,
  type AttentionFlowResponse,
  type TokenImportanceResponse,
  type TrainStepResponse,
  type LossSurfaceResponse,
  type GradientDescentResponse,
  type TrainingObjectivesResponse,
  type InferenceResponse,
  type TemperatureDemoResponse,
  type SamplingDemoResponse,
} from './client'

// ─── Hook genérico de API ───────────────────────────────────────────────────

interface UseApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

function useApiCall<T, P = void>(
  fetcher: (params: P) => Promise<T>,
  debounceMs = 300
) {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  })
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const execute = useCallback(
    (params: P) => {
      if (timerRef.current) clearTimeout(timerRef.current)
      if (abortRef.current) abortRef.current.abort()

      timerRef.current = setTimeout(async () => {
        const controller = new AbortController()
        abortRef.current = controller

        setState((prev) => ({ ...prev, loading: true, error: null }))
        try {
          const data = await fetcher(params)
          if (!controller.signal.aborted) {
            setState({ data, loading: false, error: null })
          }
        } catch (err: unknown) {
          if (controller.signal.aborted) return
          const message =
            err instanceof Error ? err.message : 'Erro ao conectar com o backend'
          setState((prev) => ({ ...prev, loading: false, error: message }))
        }
      }, debounceMs)
    },
    [fetcher, debounceMs]
  )

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return { ...state, execute, reset }
}

// ─── Hook de fetch ao montar (GET endpoints) ────────────────────────────────

function useApiFetch<T>(url: string) {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: true,
    error: null,
  })

  const refetch = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    try {
      const res = await apiClient.get<T>(url)
      setState({ data: res.data, loading: false, error: null })
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Erro ao conectar com o backend'
      setState((prev) => ({ ...prev, loading: false, error: message }))
    }
  }, [url])

  useEffect(() => {
    refetch()
  }, [refetch])

  return { ...state, refetch }
}

// ─── Tokenization ───────────────────────────────────────────────────────────

export function useTokenize() {
  return useApiCall<TokenizationResponse, { texto: string }>(
    async ({ texto }) => {
      const res = await apiClient.post<TokenizationResponse>(
        '/api/tokenization/tokenize',
        { texto }
      )
      return res.data
    }
  )
}

export function useBPESteps() {
  return useApiCall<BPEStepsResponse, { texto: string; num_merges?: number }>(
    async (params) => {
      const res = await apiClient.post<BPEStepsResponse>(
        '/api/tokenization/bpe-steps',
        params
      )
      return res.data
    }
  )
}

export function useCompareTokenizers() {
  return useApiFetch<CompareTokenizersResponse>(
    '/api/tokenization/compare-tokenizers'
  )
}

// ─── Embeddings ─────────────────────────────────────────────────────────────

export function useEmbeddings() {
  return useApiCall<EmbeddingResponse, { tokens: string[]; d_model?: number }>(
    async (params) => {
      const res = await apiClient.post<EmbeddingResponse>(
        '/api/embeddings/embeddings',
        params
      )
      return res.data
    }
  )
}

export function usePositionalEncoding() {
  return useApiCall<
    PositionalEncodingResponse,
    { seq_length?: number; d_model?: number }
  >(async (params) => {
    const res = await apiClient.post<PositionalEncodingResponse>(
      '/api/embeddings/positional-encoding',
      params
    )
    return res.data
  })
}

export function useEmbeddingSpace() {
  return useApiCall<
    EmbeddingSpaceResponse,
    { tokens: string[]; metodo?: 'pca' | 'tsne'; d_model?: number }
  >(async (params) => {
    const res = await apiClient.post<EmbeddingSpaceResponse>(
      '/api/embeddings/embedding-space',
      params
    )
    return res.data
  })
}

// ─── Attention ──────────────────────────────────────────────────────────────

export function useSelfAttention() {
  return useApiCall<
    SelfAttentionResponse,
    { tokens: string[]; d_model?: number }
  >(async (params) => {
    const res = await apiClient.post<SelfAttentionResponse>(
      '/api/attention/self-attention',
      params
    )
    return res.data
  })
}

export function useMultiHeadAttention() {
  return useApiCall<
    MultiHeadAttentionResponse,
    { tokens: string[]; d_model?: number; num_cabecas?: number }
  >(async (params) => {
    const res = await apiClient.post<MultiHeadAttentionResponse>(
      '/api/attention/multi-head-attention',
      params
    )
    return res.data
  })
}

export function useAttentionFlow() {
  return useApiCall<
    AttentionFlowResponse,
    { tokens: string[]; indice_token?: number; d_model?: number }
  >(async (params) => {
    const res = await apiClient.post<AttentionFlowResponse>(
      '/api/attention/attention-flow',
      params
    )
    return res.data
  })
}

export function useTokenImportance() {
  return useApiCall<TokenImportanceResponse, { tokens: string[]; d_model?: number }>(
    async (params) => {
      const res = await apiClient.post<TokenImportanceResponse>(
        '/api/attention/token-importance',
        params
      )
      return res.data
    }
  )
}

// ─── Training ───────────────────────────────────────────────────────────────

export function useTrainStep() {
  return useApiCall<
    TrainStepResponse,
    { taxa_aprendizado?: number; tamanho_vocab?: number; resetar?: boolean }
  >(
    async (params) => {
      const res = await apiClient.post<TrainStepResponse>(
        '/api/training/train-step',
        params
      )
      return res.data
    },
    0
  )
}

export function useLossSurface() {
  return useApiCall<LossSurfaceResponse, { resolucao?: number }>(
    async (params) => {
      const res = await apiClient.post<LossSurfaceResponse>(
        '/api/training/loss-surface',
        params
      )
      return res.data
    }
  )
}

export function useGradientDescent() {
  return useApiCall<
    GradientDescentResponse,
    {
      taxa_aprendizado?: number
      num_iteracoes?: number
      w1_inicial?: number
      w2_inicial?: number
    }
  >(async (params) => {
    const res = await apiClient.post<GradientDescentResponse>(
      '/api/training/gradient-descent-demo',
      params
    )
    return res.data
  })
}

export function useTrainingObjectives() {
  return useApiFetch<TrainingObjectivesResponse>(
    '/api/training/training-objectives'
  )
}

// ─── Inference ──────────────────────────────────────────────────────────────

export function useGenerate() {
  return useApiCall<
    InferenceResponse,
    {
      prompt?: string[]
      estrategia?: string
      max_tokens?: number
      temperatura?: number
      k?: number
      p?: number
    }
  >(
    async (params) => {
      const res = await apiClient.post<InferenceResponse>(
        '/api/inference/generate',
        params
      )
      return res.data
    },
    0
  )
}

export function useTemperatureDemo() {
  return useApiCall<TemperatureDemoResponse, { temperaturas?: number[] }>(
    async (params) => {
      const res = await apiClient.post<TemperatureDemoResponse>(
        '/api/inference/temperature-demo',
        params
      )
      return res.data
    }
  )
}

export function useSamplingDemo() {
  return useApiCall<
    SamplingDemoResponse,
    { prompt?: string[]; max_tokens?: number; num_beams?: number }
  >(async (params) => {
    const res = await apiClient.post<SamplingDemoResponse>(
      '/api/inference/sampling-demo',
      params
    )
    return res.data
  })
}

