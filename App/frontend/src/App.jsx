import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { supabase } from './lib/supabaseClient'; // Import du client configuré
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import CapturePage from './pages/CapturePage';
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import Stats from "./pages/Stats";


function App() {
  
  // Test de connexion automatique au lancement
  useEffect(() => {
    const checkConnection = async () => {
      try {
        // On essaie de lire la table agents (même si elle est vide)
        const { data, error } = await supabase.from('agents').select('id').limit(1);
        
        if (error) {
          console.warn("⚠️ Statut Supabase :", error.message);
        } else {
          console.log("✅ Supabase est connecté avec succès !");
        }
      } catch (err) {
        console.error("❌ Erreur critique de configuration :", err);
      }
    };

    checkConnection();
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/capture" element={<CapturePage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/stats" element={<Stats />} />

      </Routes>
    </Router>
  );
}

export default App;