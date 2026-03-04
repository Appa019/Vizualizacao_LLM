import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
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
import LandingPage from './pages/LandingPage'

export default function App() {
  return (
    <Routes>
      <Route path="/home" element={<LandingPage />} />
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
