import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'
import Layout from './componentes/Layout'
import Dashboard from './paginas/Dashboard'
import Datasets from './paginas/Datasets'
import Limpieza from './paginas/Limpieza'
import Entrenamiento from './paginas/Entrenamiento'
import Resultados from './paginas/Resultados'
import Prediccion from './paginas/Prediccion'
import AnalisisAvanzado from './paginas/AnalisisAvanzado'
import AnalisisMercado from './paginas/AnalisisMercado'
import Login from './paginas/Login'

function App() {
  const { usuario, cargando } = useAuth()

  if (cargando) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 to-primary-100">
        <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <Routes>
      {/* RUTA DE LOGIN */}
      <Route
        path="/login"
        element={
          !usuario ? <Login /> : <Navigate to="/" replace />
        }
      />

      {/* RUTAS PRIVADAS (solo accesibles si hay usuario autenticado) */}
      <Route
        path="/"
        element={
          usuario ? (
            <Layout>
              <Dashboard />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/datasets"
        element={
          usuario ? (
            <Layout>
              <Datasets />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/limpieza/:id"
        element={
          usuario ? (
            <Layout>
              <Limpieza />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/entrenamiento"
        element={
          usuario ? (
            <Layout>
              <Entrenamiento />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/resultados"
        element={
          usuario ? (
            <Layout>
              <Resultados />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/prediccion"
        element={
          usuario ? (
            <Layout>
              <Prediccion />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/analisis-avanzado"
        element={
          usuario ? (
            <Layout>
              <AnalisisAvanzado />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/analisis-mercado"
        element={
          usuario ? (
            <Layout>
              <AnalisisMercado />
            </Layout>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />

      {/* Redirección por defecto */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
