# LLM Explorer — Plataforma Educacional de Transformers

Plataforma interativa para aprender **tudo** sobre Large Language Models: do treinamento a inferencia, com visualizacoes 3D, formulas matematicas e explicacoes passo a passo.

## Modulos

| # | Modulo | O que ensina |
|---|--------|-------------|
| 1 | O que sao LLMs | Historia, timeline RNN→LSTM→Transformer, fundamentos |
| 2 | Tokenizacao | BPE interativo, comparacao de tokenizadores |
| 3 | Embeddings | Espaco vetorial 3D, positional encoding, distancia coseno |
| 4 | Arquitetura | Transformer completo: FFN, LayerNorm, residual connections |
| 5 | Atencao | Self-Attention passo a passo, Multi-Head, heatmaps 3D |
| 6 | Treinamento | Loss surface 3D, gradient descent, mini-modelo treinavel |
| 7 | Inferencia | Temperature, top-k, top-p, beam search, geracao ao vivo |
| 8 | Fine-tuning | RLHF, LoRA, transfer learning |
| 9 | Laboratorio | Pipeline completo interativo, comparador de modelos |

## Stack

- **Backend**: Python 3.11+, FastAPI, NumPy, SciPy, scikit-learn, HuggingFace Transformers, PyTorch
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Three.js (R3F), Plotly.js, KaTeX, Framer Motion

## Quick Start

```bash
# Tudo de uma vez
./start.sh

# Ou separadamente:

# Backend
cd backend
source ../.venv/bin/activate
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install   # primeira vez
npm run dev
```

Acesse:
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

## Estrutura

```
backend/
  main.py           # FastAPI + CORS + routers
  core/             # Logica de dominio (simulador, trainer, gerador)
  routers/          # Endpoints REST (tokenization, embeddings, attention, training, inference, models)
  utils/            # Tokenizador, math helpers
frontend/
  src/
    pages/          # 10 paginas educacionais
    components/
      viz/          # EmbeddingSpace 3D, Heatmap3D, LossSurface, AttentionFlow, NetworkGraph
      education/    # EducationalViz, FormulaBlock (KaTeX), StepByStep, WhyItMatters
      layout/       # Sidebar, Header, Layout
      ui/           # Card, Toggle, Slider, StepIndicator
    api/client.ts   # Axios client
```

## Principio Educativo

Toda visualizacao segue o padrao **"Veja, Entenda, Aplique"**:

1. **Legenda**: explica o que o grafico mostra em linguagem simples
2. **Formula**: renderizada com KaTeX, variaveis clicaveis com cores consistentes (Q=vermelho, K=verde, V=azul)
3. **"Por que importa?"**: conexao com uso pratico real
4. **Interatividade**: rotacao 3D, hover com detalhes, sliders para parametros
