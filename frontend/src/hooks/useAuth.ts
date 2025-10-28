import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import type { User } from '@supabase/supabase-js'

export function useAuth() {
  const [usuario, setUsuario] = useState<User | null>(null)
  const [cargando, setCargando] = useState(true)

  useEffect(() => {
    const savedUser = localStorage.getItem('usuario_local')
    if (savedUser) {
      setUsuario(JSON.parse(savedUser))
      setCargando(false)
      return
    }

    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUsuario(session.user)
      }
      setCargando(false)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setUsuario(session?.user ?? null)
      if (session?.user) {
        localStorage.setItem('usuario_local', JSON.stringify(session.user))
      } else {
        localStorage.removeItem('usuario_local')
      }
    })

    return () => subscription.unsubscribe()
  }, [])

  const iniciarSesion = async (email: string, password: string) => {
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    return { error }
  }

  const registrarse = async (email: string, password: string) => {
    const { error } = await supabase.auth.signUp({ email, password })
    return { error }
  }

  const cerrarSesion = async () => {
    await supabase.auth.signOut()
    localStorage.removeItem('usuario_local')
    setUsuario(null)
    window.location.href = '/login'
  }

  return { usuario, cargando, iniciarSesion, registrarse, cerrarSesion }
}