import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'

// ─── Páginas ──────────────────────────────────────────────────────────────────
import Home from './pages/Home'
import WhatAreLLMs from './pages/WhatAreLLMs'
import Tokenization from './pages/Tokenization'
import Embeddings from './pages/Embeddings'
import Architecture from './pages/Architecture'
import Attention from './pages/Attention'
import Training from './pages/Training'
import Inference from './pages/Inference'
import FineTuning from './pages/FineTuning'
import Lab from './pages/Lab'

// ─── App raiz com roteamento ──────────────────────────────────────────────────

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index path="/" element={<Home />} />
        <Route path="/what-are-llms" element={<WhatAreLLMs />} />
        <Route path="/tokenization" element={<Tokenization />} />
        <Route path="/embeddings" element={<Embeddings />} />
        <Route path="/architecture" element={<Architecture />} />
        <Route path="/attention" element={<Attention />} />
        <Route path="/training" element={<Training />} />
        <Route path="/inference" element={<Inference />} />
        <Route path="/fine-tuning" element={<FineTuning />} />
        <Route path="/lab" element={<Lab />} />
      </Route>
    </Routes>
  )
}
