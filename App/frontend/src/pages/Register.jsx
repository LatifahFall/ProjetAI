import React, { useState } from 'react';
import { Brain, User, Mail, Lock, Briefcase, ArrowRight } from 'lucide-react';

const Register = () => {
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    company: '',
    password: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Données d'inscription :", formData);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 px-6 lg:px-8 relative">
      
      {/* Retour à la connexion */}
      <div className="absolute top-8 left-8">
        <button 
          onClick={() => window.location.href = '/login'}
          className="flex items-center gap-2 text-sm font-medium text-gray-500 hover:text-[#646cff] transition-colors"
        >
          ← Retour à la connexion
        </button>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 bg-gradient-ocean rounded-2xl flex items-center justify-center shadow-lg shadow-indigo-200">
            <Brain size={32} className="text-white" />
          </div>
        </div>
        <h2 className="text-center text-3xl font-black tracking-tight text-gray-900">
          Inscription Entreprise
        </h2>
        <p className="mt-2 text-center text-sm text-gray-500 font-medium">
          Créez votre espace de prédiction de personnalité
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-10 px-8 shadow-2xl shadow-indigo-100 rounded-[2.5rem] border border-gray-100">
          <form className="space-y-5" onSubmit={handleSubmit}>
            
            {/* Nom Complet */}
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2 ml-1">
                Nom Complet
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-400">
                  <User size={18} />
                </div>
                <input
                  type="text"
                  required
                  className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm focus:outline-none focus:ring-2 focus:ring-[#646cff] focus:bg-white transition-all"
                  placeholder="Jean Dupont"
                  onChange={(e) => setFormData({...formData, fullName: e.target.value})}
                />
              </div>
            </div>

            {/* Email Professionnel */}
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2 ml-1">
                Email Professionnel
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-400">
                  <Mail size={18} />
                </div>
                <input
                  type="email"
                  required
                  className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm focus:outline-none focus:ring-2 focus:ring-[#646cff] focus:bg-white transition-all"
                  placeholder="jean@entreprise.com"
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                />
              </div>
            </div>

            {/* Entreprise */}
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2 ml-1">
                Nom de l'entreprise
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-400">
                  <Briefcase size={18} />
                </div>
                <input
                  type="text"
                  required
                  className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm focus:outline-none focus:ring-2 focus:ring-[#646cff] focus:bg-white transition-all"
                  placeholder="Ocean Insight Inc."
                  onChange={(e) => setFormData({...formData, company: e.target.value})}
                />
              </div>
            </div>

            {/* Mot de passe */}
            <div>
              <label className="block text-xs font-black uppercase tracking-widest text-gray-400 mb-2 ml-1">
                Définir un mot de passe
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none text-gray-400">
                  <Lock size={18} />
                </div>
                <input
                  type="password"
                  required
                  className="block w-full pl-11 pr-4 py-4 bg-gray-50 border border-gray-100 rounded-2xl text-sm focus:outline-none focus:ring-2 focus:ring-[#646cff] focus:bg-white transition-all"
                  placeholder="••••••••"
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full mt-6 flex justify-center items-center gap-2 py-4 px-4 rounded-2xl shadow-lg text-sm font-black text-white bg-gradient-ocean hover:scale-[1.02] active:scale-95 transition-all"
            >
              Créer mon compte pro
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
        
        <p className="mt-8 text-center text-xs text-gray-400 font-medium">
          Déjà inscrit ? <a href="/login" className="text-[#646cff] font-bold hover:underline">Se connecter</a>
        </p>
      </div>
    </div>
  );
};

export default Register;