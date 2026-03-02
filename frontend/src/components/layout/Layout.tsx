import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'

// ─── Layout principal da aplicação ───────────────────────────────────────────
// Estrutura: sidebar fixa à esquerda + área de conteúdo com header sticky

export interface LayoutContext {
  modoSimulacao: boolean
}

export default function Layout() {
  const [modoSimulacao, setModoSimulacao] = useState(true)

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      {/* Sidebar de navegação */}
      <Sidebar />

      {/* Área de conteúdo principal */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {/* Header sticky com título e controles */}
        <Header
          modoSimulacao={modoSimulacao}
          onAlternarModo={setModoSimulacao}
        />

        {/* Conteúdo da rota atual */}
        <main className="flex-1 overflow-y-auto bg-white">
          <div className="animate-fade-in">
            <Outlet context={{ modoSimulacao }} />
          </div>
        </main>
      </div>
    </div>
  )
}
