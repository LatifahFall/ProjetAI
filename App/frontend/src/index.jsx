import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // Vérifie que App.jsx existe bien
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);