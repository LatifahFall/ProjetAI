import React, { useState } from 'react';
import { Brain, Lock, Mail, ArrowRight } from 'lucide-react';
import { supabase } from '../lib/supabaseClient';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({ email: '', password: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // On cherche l'agent qui a cet email et ce mot de passe
      const { data, error } = await supabase
        .from('agents')
        .select('*')
        .eq('email', formData.email)
        .eq('mot_de_passe', formData.password)
        .single(); // On attend un seul résultat

      if (error || !data) {
        throw new Error("Email ou mot de passe incorrect");
      }

      console.log("Connecté avec succès :", data);
      alert(`Bienvenue ${data.nom_complet} !`);
      
      // On stocke l'ID de l'agent dans le navigateur pour s'en souvenir
      localStorage.setItem('agent_id', data.id);
      
      navigate('/dashboard'); // Retour à l'accueil ou vers un Dashboard
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 px-6 lg:px-8 relative">
      <div className="absolute top-8 left-8">
        <button onClick={() => navigate('/')} className="flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-[#646cff] transition-colors">
          ← Retour à l'accueil
        </button>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 bg-[#646cff] rounded-2xl flex items-center justify-center shadow-lg">
            <Brain size={32} className="text-white" />
          </div>
        </div>
        <h2 className="text-center text-3xl font-black tracking-tight text-gray-900">Espace Agent</h2>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-10 px-8 shadow-2xl rounded-[2.5rem] border border-gray-100">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2">Email Professionnel</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Mail size={18} /></div>
                <input type="email" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm" placeholder="agent@entreprise.ai"
                  onChange={(e) => setFormData({...formData, email: e.target.value})} />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2">Mot de passe</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Lock size={18} /></div>
                <input type="password" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm" placeholder="••••••••"
                  onChange={(e) => setFormData({...formData, password: e.target.value})} />
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full flex justify-center items-center gap-2 py-4 px-4 rounded-2xl shadow-lg text-sm font-black text-white bg-[#646cff]">
              {loading ? "Connexion..." : "Se connecter"}
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
        <p className="mt-8 text-center text-xs text-gray-400">
          Nouveau ? <button onClick={() => navigate('/register')} className="text-[#646cff] font-bold">Créer un compte</button>
        </p>
      </div>
    </div>
  );
};

export default Login;