// src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import CreditScoringApp from './App.jsx'; // ← This is correct even if file is App.js

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <CreditScoringApp />
  </React.StrictMode>
);
