import React, { useState } from 'react';
import { Brain, Lock, Mail, ArrowRight, ArrowLeft } from 'lucide-react';
import { supabase } from '../lib/supabaseClient';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [formData, setFormData] = useState({ email: '', password: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg('');

    try {
      const { data, error } = await supabase
        .from('agents')
        .select('*')
        .eq('email', formData.email)
        .eq('mot_de_passe', formData.password)
        .single();

      if (error || !data) throw new Error("Identifiants incorrects");

      localStorage.setItem('agent_id', data.id);
      navigate('/dashboard'); 
    } catch (error) {
      setErrorMsg(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 px-6 relative">
      <div className="absolute top-8 left-8">
        <button onClick={() => navigate('/')} className="flex items-center gap-2 text-sm font-bold text-gray-400 hover:text-[#646cff]">
          <ArrowLeft size={16} /> Accueil
        </button>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <div className="inline-flex w-16 h-16 bg-[#646cff] rounded-2xl items-center justify-center shadow-lg mb-6">
          <Brain size={32} className="text-white" />
        </div>
        <h2 className="text-3xl font-black text-gray-900">Espace Agent</h2>
        {errorMsg && <p className="mt-4 text-red-500 text-sm font-bold bg-red-50 py-2 rounded-xl">{errorMsg}</p>}
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-10 px-8 shadow-2xl rounded-[2.5rem] border border-gray-100">
          <form className="space-y-6" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-black uppercase text-gray-400 mb-2 ml-1">Email</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Mail size={18} /></div>
                <input type="email" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-2xl text-sm focus:ring-2 focus:ring-[#646cff]" placeholder="agent@entreprise.ai"
                  onChange={(e) => setFormData({...formData, email: e.target.value})} />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black uppercase text-gray-400 mb-2 ml-1">Mot de passe</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Lock size={18} /></div>
                <input type="password" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-2xl text-sm focus:ring-2 focus:ring-[#646cff]" placeholder="••••••••"
                  onChange={(e) => setFormData({...formData, password: e.target.value})} />
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full flex justify-center items-center gap-2 py-4 px-4 rounded-2xl shadow-lg text-sm font-black text-white bg-[#646cff] hover:opacity-90 transition-all">
              {loading ? "Chargement..." : "Se connecter"}
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
        <p className="mt-8 text-center text-xs text-gray-400 font-bold">
          Pas de compte ? <button onClick={() => navigate('/register')} className="text-[#646cff]">S'inscrire</button>
        </p>
      </div>
    </div>
  );
};

export default Login;