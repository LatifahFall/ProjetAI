import React, { useState } from 'react';
import { Brain, User, Mail, Lock, Briefcase, ArrowRight, ArrowLeft } from 'lucide-react';
import { supabase } from '../lib/supabaseClient';
import { useNavigate } from 'react-router-dom';

const Register = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({ fullName: '', email: '', company: '', password: '' });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const { error } = await supabase.from('agents').insert([{ 
          nom_complet: formData.fullName, 
          email: formData.email, 
          mot_de_passe: formData.password, 
          entreprise_nom: formData.company 
      }]);
      if (error) throw error;
      navigate('/login'); 
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 px-6 relative">
      <div className="absolute top-8 left-8">
        <button onClick={() => navigate('/login')} className="flex items-center gap-2 text-sm font-bold text-gray-400 hover:text-[#646cff]">
          <ArrowLeft size={16} /> Retour
        </button>
      </div>
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center mb-8">
        <div className="inline-flex w-16 h-16 bg-[#646cff] rounded-2xl items-center justify-center shadow-lg mb-4">
          <Brain size={32} className="text-white" />
        </div>
        <h2 className="text-3xl font-black text-gray-900">Nouveau Compte Pro</h2>
      </div>

      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-10 px-8 shadow-2xl rounded-[2.5rem] border border-gray-100">
          <form className="space-y-5" onSubmit={handleSubmit}>
            <InputGroup label="Nom Complet" icon={<User size={18}/>} placeholder="Jean Dupont" onChange={(v) => setFormData({...formData, fullName: v})} />
            <InputGroup label="Email Pro" icon={<Mail size={18}/>} placeholder="jean@entreprise.com" type="email" onChange={(v) => setFormData({...formData, email: v})} />
            <InputGroup label="Entreprise" icon={<Briefcase size={18}/>} placeholder="Nom de la société" onChange={(v) => setFormData({...formData, company: v})} />
            <InputGroup label="Mot de passe" icon={<Lock size={18}/>} placeholder="••••••••" type="password" onChange={(v) => setFormData({...formData, password: v})} />

            <button type="submit" disabled={loading} className="w-full mt-6 flex justify-center items-center gap-2 py-4 px-4 rounded-2xl shadow-lg text-sm font-black text-white bg-[#646cff] hover:scale-[1.02] transition-all">
              {loading ? "Création..." : "Créer mon compte"}
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

const InputGroup = ({ label, icon, placeholder, type = "text", onChange }) => (
  <div>
    <label className="block text-xs font-black uppercase text-gray-400 mb-2 ml-1">{label}</label>
    <div className="relative">
      <div className="absolute inset-y-0 left-0 pl-4 flex items-center text-gray-400">{icon}</div>
      <input type={type} required className="block w-full pl-11 pr-4 py-4 bg-gray-50 border-none rounded-2xl text-sm focus:ring-2 focus:ring-[#646cff]" 
      placeholder={placeholder} onChange={(e) => onChange(e.target.value)} />
    </div>
  </div>
);

export default Register;