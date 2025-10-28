import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { FacialAuthOverlay } from "../componentes/FacialAuthOverlay";

export default function Login() {
  const [fullName, setFullName] = useState("");
  const [isFacialMode, setIsFacialMode] = useState(false);
  const [esRegistro, setEsRegistro] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleFacialSuccess = async (userData: any) => {
    console.log("✅ Autenticación facial exitosa:", userData);

    if (userData.token) {
      try {
        const { error } = await supabase.auth.setSession({
          access_token: userData.token,
          refresh_token: userData.token,
        });

        if (error) {
          console.error('Error al aplicar sesión facial:', error);
        }

        localStorage.setItem('usuario_local', JSON.stringify({
          id: userData.user_id,
          email: `${userData.full_name}@facial.local`,
          name: userData.full_name
        }));
      } catch (err) {
        console.error('Error al guardar sesión:', err);
      }
    }

    setTimeout(() => {
      navigate("/");
      window.location.reload();
    }, 500);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-600 via-purple-600 to-pink-600 p-4 relative overflow-hidden">
      <div className="absolute inset-0 bg-black/20"></div>

      <div className="relative w-full max-w-md animate-scale-in">
        <div className="glass border border-white/20 rounded-3xl shadow-2xl p-8">
          <div className="text-center mb-8">
            <div className="relative w-20 h-20 mx-auto mb-6">
              <div className="absolute inset-0 bg-gradient-to-br from-white to-blue-100 rounded-3xl rotate-6"></div>
              <div className="relative w-full h-full bg-gradient-to-br from-primary-600 to-purple-600 rounded-3xl flex items-center justify-center shadow-xl">
                <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
            </div>

            <h2 className="text-4xl font-bold text-white mb-2">
              {esRegistro ? "Registro Facial" : "Acceso Biométrico"}
            </h2>
            <p className="text-white/90 text-lg">
              {esRegistro ? "Regístrese con su rostro" : "Verifique su identidad"}
            </p>
          </div>

          {isFacialMode ? (
            <FacialAuthOverlay
              mode={esRegistro ? "register" : "login"}
              fullName={fullName}
              onSuccess={handleFacialSuccess}
              onCancel={() => setIsFacialMode(false)}
            />
          ) : (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                setError("");
                setIsFacialMode(true);
              }}
              className="space-y-5"
            >
              <div>
                <label className="block text-sm font-semibold text-white mb-2">
                  Nombre Completo
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 10-8 0 4 4 0 008 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <input
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="input-field pl-12"
                    placeholder="Su Nombre Completo"
                    required
                  />
                </div>
              </div>

              {error && (
                <div className="bg-rose-500/20 border-2 border-rose-400 text-white px-4 py-3 rounded-xl text-sm font-medium backdrop-blur-sm flex items-center space-x-2">
                  <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}

              <button
                type="submit"
                className="w-full bg-white text-primary-600 py-4 rounded-xl font-bold text-lg hover:bg-slate-50 transition-all duration-200 shadow-lg hover:shadow-xl hover:scale-105"
              >
                {esRegistro ? "Registrarse con Rostro" : "Iniciar Sesión con Rostro"}
              </button>
            </form>
          )}

          <div className="mt-6 text-center">
            <button
              onClick={() => {
                setEsRegistro(!esRegistro);
                setIsFacialMode(false);
                setFullName("");
                setError("");
              }}
              className="text-white hover:text-white/90 text-sm font-semibold transition-colors underline decoration-2 underline-offset-4"
            >
              {esRegistro
                ? "¿Ya tienes cuenta? Inicia sesión"
                : "¿No tienes cuenta? Regístrate"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}