import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
// 1. Importamos BrowserRouter
import { BrowserRouter } from 'react-router-dom';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* 2. Envolvemos <App /> en BrowserRouter */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
