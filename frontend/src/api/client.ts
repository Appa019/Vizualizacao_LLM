import axios from 'axios'

// Cliente HTTP configurado para a API backend em localhost:8000
const apiClient = axios.create({
  baseURL: '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Interceptor de requisição — adiciona headers comuns
apiClient.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Interceptor de resposta — tratamento centralizado de erros
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 503) {
      console.warn('[API] Serviço indisponível — verifique se o backend está rodando.')
    }
    return Promise.reject(error)
  }
)

export default apiClient

// ─── Tipos base ─────────────────────────────────────────────────────────────

export interface ApiError {
  detail: string
  status: number
}

// ─── Tokenization ───────────────────────────────────────────────────────────

export interface TokenizationRequest {
  texto: string
  modelo?: string
}

export interface TokenizationResponse {
  tokens: string[]
  ids: number[]
  contagem: number
}

export interface BPEStepsRequest {
  texto: string
  num_merges?: number
}

export interface BPEStep {
  passo: number
  par: [string, string]
  frequencia: number
  tokens_atuais: string[]
  vocabulario_tamanho: number
}

export interface BPEStepsResponse {
  texto_original: string
  tokens_iniciais: string[]
  passos: BPEStep[]
  tokens_finais: string[]
  estatisticas: {
    tokens_iniciais: number
    tokens_finais: number
    reducao_percentual: number
  }
}

export interface CompareTokenizersResponse {
  texto: string
  abordagens: {
    nome: string
    tokens: string[]
    num_tokens: number
    descricao: string
  }[]
}

// ─── Embeddings ─────────────────────────────────────────────────────────────

export interface EmbeddingRequest {
  tokens: string[]
  d_model?: number
}

export interface EmbeddingResponse {
  embeddings: number[][]
  positional_encoding: number[][]
  embeddings_finais: number[][]
  tokens: string[]
  d_model: number
  num_tokens_reais: number
  dimensoes: number
}

export interface PositionalEncodingRequest {
  seq_length?: number
  d_model?: number
}

export interface PositionalEncodingResponse {
  encoding: number[][]
  seq_length: number
  d_model: number
  explicacao: string
}

export interface EmbeddingSpaceRequest {
  tokens: string[]
  metodo?: 'pca' | 'tsne'
  d_model?: number
}

export interface EmbeddingSpacePoint {
  word: string
  position: [number, number, number]
  category: string
}

export interface EmbeddingSpaceResponse {
  pontos: EmbeddingSpacePoint[]
  metodo: string
  variancia_explicada?: number[]
  explicacao: string
}

// ─── Attention ──────────────────────────────────────────────────────────────

export interface AttentionRequest {
  tokens: string[]
  d_model?: number
  camada?: number
  cabeca?: number
}

export interface SelfAttentionResponse {
  Q: number[][]
  K: number[][]
  V: number[][]
  scores_escalados: number[][]
  pesos_atencao: number[][]
  saida: number[][]
  tokens: string[]
  num_tokens_reais: number
  fator_escala: number
  passos_explicados: { passo: string; descricao: string }[]
}

export interface MultiHeadAttentionRequest {
  tokens: string[]
  d_model?: number
  num_cabecas?: number
}

export interface MultiHeadAttentionResponse {
  cabecas: {
    cabeca: number
    tokens: string[]
    pesos_atencao: number[][]
  }[]
  saida_final: number[][]
  tokens: string[]
  num_tokens_reais: number
  d_model: number
  num_cabecas: number
  d_k: number
  explicacao: string
}

export interface AttentionFlowRequest {
  tokens: string[]
  indice_token?: number
  d_model?: number
}

export interface AttentionFlowResponse {
  token_alvo: string
  indice_token: number
  pesos: number[]
  tokens: string[]
  conexoes_significativas: {
    token: string
    indice: number
    peso: number
  }[]
  explicacao: string
}

export interface TokenImportanceRequest {
  tokens: string[]
  d_model?: number
}

export interface TokenImportanceResponse {
  tokens: string[]
  importancia_recebida: number[]
  importancia_dada: number[]
  importancia_combinada: number[]
  token_mais_importante: string
  explicacao: string
}

export interface AttentionResponse {
  pesos: number[][]
  tokens: string[]
  camada: number
  cabeca: number
}

// ─── Training ───────────────────────────────────────────────────────────────

export interface TrainStepRequest {
  taxa_aprendizado?: number
  tamanho_vocab?: number
  resetar?: boolean
}

export interface TrainStepResponse {
  passo: number
  loss: number
  acuracia: number
  gradientes_norma: Record<string, number>
  historico_loss: number[]
  predicao_top5: {
    indice: number
    token: string
    probabilidade: number
    correto: boolean
  }[]
  alvo: number
  explicacao: string
}

export interface LossSurfaceRequest {
  resolucao?: number
}

export interface LossSurfaceResponse {
  w1_valores: number[]
  w2_valores: number[]
  loss_grid: number[][]
  ponto_otimo: { w1: number; w2: number; loss: number }
  explicacao: string
}

export interface GradientDescentRequest {
  taxa_aprendizado?: number
  num_iteracoes?: number
  w1_inicial?: number
  w2_inicial?: number
}

export interface GradientDescentResponse {
  passos: {
    iteracao: number
    w1: number
    w2: number
    loss: number
    gradiente_w1: number
    gradiente_w2: number
  }[]
  taxa_aprendizado: number
  loss_inicial: number
  loss_final: number
  convergiu: boolean
  reducao_percentual: number
  explicacao: string
}

export interface TrainingObjectivesResponse {
  frase_original: string
  tokens: string[]
  mlm: Record<string, unknown>
  clm: Record<string, unknown>
  comparacao: string
}

// ─── Inference ──────────────────────────────────────────────────────────────

export interface InferenceRequest {
  prompt?: string[]
  estrategia?: string
  max_tokens?: number
  temperatura?: number
  k?: number
  p?: number
}

export interface TokenGerado {
  token: string
  probabilidade: number
  logit: number
  top_5_tokens: { token: string; probabilidade: number }[]
}

export interface InferenceResponse {
  texto_gerado: string
  tokens_gerados: string[]
  historico_probabilidades: TokenGerado[]
  estrategia: string
  parametros: Record<string, unknown>
  explicacao: string
}

export interface TemperatureDemoRequest {
  temperaturas?: number[]
}

export interface TemperatureDemoResponse {
  logits_originais: number[]
  tokens: string[]
  distribuicoes: {
    temperatura: number
    probabilidades: number[]
    entropia: number
    descricao: string
  }[]
  explicacao: string
}

export interface SamplingDemoRequest {
  prompt?: string[]
  max_tokens?: number
  num_beams?: number
}

export interface SamplingDemoResponse {
  greedy: Record<string, unknown>
  temperatura_baixa: Record<string, unknown>
  temperatura_alta: Record<string, unknown>
  top_k: Record<string, unknown>
  top_p: Record<string, unknown>
  beam_search: Record<string, unknown>[]
  explicacao: string
}

