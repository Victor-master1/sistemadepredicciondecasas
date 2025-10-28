import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
// CORRECCIÓN: Importamos la clase FaceMesh a través de un alias de módulo
import * as faceMeshModule from '@mediapipe/face_mesh'; 
import { FACEMESH_TESSELATION } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

// Define la URL base de la API desde las variables de entorno de Vercel
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

interface FacialAuthOverlayProps {
  mode: 'register' | 'login';
  fullName: string;
  onSuccess: (userData: any) => void;
  onCancel: () => void;
}

export const FacialAuthOverlay: React.FC<FacialAuthOverlayProps> = ({
  mode,
  fullName,
  onSuccess,
  onCancel,
}) => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [faceMeshReady, setFaceMeshReady] = useState(false);

  useEffect(() => {
    let camera: Camera | null = null;

    const initializeFaceMesh = async () => {
      if (!webcamRef.current?.video) return;

      // CORRECCIÓN CLAVE: Usamos el alias faceMeshModule.FaceMesh para evitar el TypeError
      const faceMesh = new faceMeshModule.FaceMesh({ 
        locateFile: (file) => {
          // Solución FINAL para la carga de assets: usar la ruta oficial de JS Deliver
          // con la versión explícita.
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`;
        },
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      faceMesh.onResults(onResults);

      if (webcamRef.current.video) {
        camera = new Camera(webcamRef.current.video, {
          onFrame: async () => {
            if (webcamRef.current?.video) {
              await faceMesh.send({ image: webcamRef.current.video });
            }
          },
          width: 640,
          height: 480,
        });

        camera.start();
        setFaceMeshReady(true);
      }
    };

    // Usamos setTimeout para dar tiempo al navegador para inicializar
    const timeoutId = setTimeout(initializeFaceMesh, 1000); 

    return () => {
      clearTimeout(timeoutId);
      if (camera) {
        camera.stop();
      }
    };
  }, []);

  const onResults = (results: any) => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 640;
    canvas.height = 480;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      for (const landmarks of results.multiFaceLandmarks) {
        drawMesh(ctx, landmarks);
      }
    }

    ctx.restore();
  };

  const drawMesh = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 0.5;
    ctx.fillStyle = '#94a3b8';

    for (const connection of FACEMESH_TESSELATION) {
      const [startIdx, endIdx] = connection;
      const start = landmarks[startIdx];
      const end = landmarks[endIdx];

      if (start && end) {
        ctx.beginPath();
        ctx.moveTo(start.x * 640, start.y * 480);
        ctx.lineTo(end.x * 640, end.y * 480);
        ctx.stroke();
      }
    }

    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * 640, landmark.y * 480, 0.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  const captureImage = async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      setStatus('No se pudo capturar la imagen.');
      return;
    }

    setLoading(true);
    setStatus('Analizando rostro...');

    try {
      // Uso de la variable de entorno para la URL del backend
      const response = await axios.post(`${API_BASE_URL}/api/facial_auth`, {
        mode: mode,
        full_name: fullName,
        image_base64: imageSrc,
      });

      if (response.status === 200 || response.status === 201) {
        const data = response.data;
        setStatus(`${data.message} ✅`);
        
        setTimeout(() => {
          onSuccess({
            token: data.token,
            user_id: data.user_id,
            full_name: data.full_name || fullName,
          });
        }, 1000);
      }
    } catch (error: any) {
      console.error('Error en autenticación facial:', error);
      const errorMessage = error.response?.data?.message || 'Error al conectar con el servidor o fallo de red.';
      setStatus(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-2xl p-6 w-[680px] text-center relative">
        <h2 className="text-lg font-semibold text-gray-800 mb-2">
          {mode === 'register' ? 'Registro Facial' : 'Autenticación Facial'}
        </h2>
        <p className="text-sm text-gray-500 mb-4">
          Asegúrate de estar bien iluminado y mira de frente a la cámara.
        </p>

        <div className="relative rounded-xl overflow-hidden shadow-lg border border-gray-300">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="w-full h-[480px] object-cover"
            videoConstraints={{ 
              facingMode: 'user',
              width: 640,
              height: 480
            }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
        </div>

        {faceMeshReady && (
          <div className="mt-2 flex items-center justify-center text-xs text-green-600">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse"></div>
            Detección facial activa
          </div>
        )}

        <div className="mt-4 flex flex-col gap-2">
          <button
            onClick={captureImage}
            disabled={loading || !faceMeshReady}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Verificando...' : mode === 'register' ? 'Registrar Rostro' : 'Verificar Rostro'}
          </button>

          <button
            onClick={onCancel}
            disabled={loading}
            className="px-4 py-2 bg-gray-300 hover:bg-gray-400 text-gray-800 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancelar
          </button>
        </div>

        {status && (
          <p className="mt-3 text-sm text-gray-600 font-medium">{status}</p>
        )}
      </div>
    </div>
  );
};
