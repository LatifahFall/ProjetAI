import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Assure-toi que "root" correspond bien à l'ID dans ton index.html
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);