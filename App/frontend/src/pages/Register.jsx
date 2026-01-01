import React, { useState } from 'react';
import { Brain, User, Mail, Lock, Briefcase, ArrowRight } from 'lucide-react';
import { supabase } from '../lib/supabaseClient'; // On importe le client
import { useNavigate } from 'react-router-dom';

const Register = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    company: '',
    password: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // 1. On insère l'agent dans la table 'agents' de Supabase
      const { data, error } = await supabase
        .from('agents')
        .insert([
          { 
            nom_complet: formData.fullName, 
            email: formData.email, 
            mot_de_passe: formData.password, // En production, il faudra hasher ce MDP
            entreprise_nom: formData.company 
          }
        ]);

      if (error) throw error;

      alert("Inscription réussie ! Vous pouvez vous connecter.");
      navigate('/login'); // Redirection vers le login
    } catch (error) {
      alert("Erreur lors de l'inscription : " + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 px-6 lg:px-8 relative">
      <div className="absolute top-8 left-8">
        <button onClick={() => navigate('/login')} className="flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-[#646cff] transition-colors">
          ← Retour à la connexion
        </button>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 bg-[#646cff] rounded-2xl flex items-center justify-center shadow-lg">
            <Brain size={32} className="text-white" />
          </div>
        </div>
        <h2 className="text-center text-3xl font-black tracking-tight text-gray-900">Inscription Entreprise</h2>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-10 px-8 shadow-2xl rounded-[2.5rem] border border-gray-100">
          <form className="space-y-5" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2">Nom Complet</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><User size={18} /></div>
                <input type="text" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm" placeholder="Jean Dupont"
                  onChange={(e) => setFormData({...formData, fullName: e.target.value})} />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2">Email Professionnel</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Mail size={18} /></div>
                <input type="email" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm" placeholder="jean@entreprise.com"
                  onChange={(e) => setFormData({...formData, email: e.target.value})} />
              </div>
            </div>

            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2">Nom de l'entreprise</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400"><Briefcase size={18} /></div>
                <input type="text" required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm" placeholder="Ocean Insight Inc."
                  onChange={(e) => setFormData({...formData, company: e.target.value})} />
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

            <button type="submit" disabled={loading} className="w-full mt-6 flex justify-center items-center gap-2 py-4 px-4 rounded-2xl shadow-lg text-sm font-black text-white bg-[#646cff] hover:scale-[1.02] active:scale-95 transition-all">
              {loading ? "Chargement..." : "Créer mon compte pro"}
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Register;