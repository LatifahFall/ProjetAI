import React, { useEffect } from "react";
import { Mic, History, BarChart3, Users, LogOut, Home } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  useEffect(() => {
    if (!localStorage.getItem("agent_id")) navigate("/login");
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem("agent_id");
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gray-50 px-6 lg:px-20 py-12">
      <div className="flex justify-between items-center mb-12">
        <h1 className="text-4xl font-black text-gray-900">Dashboard</h1>
        <div className="flex gap-4">
            <button onClick={() => navigate('/')} className="p-3 bg-white rounded-xl shadow-sm text-gray-500 hover:text-blue-600"><Home size={20}/></button>
            <button onClick={handleLogout} className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-600 rounded-xl font-bold hover:bg-red-100 transition-all">
                <LogOut size={18} /> Quitter
            </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card icon={<Users size={32} />} title="Clients" desc="Gestion de la base" onClick={() => navigate("/clients")} />
        <Card icon={<Mic size={32} />} title="Analyse" desc="Nouvel enregistrement" onClick={() => navigate("/capture")} />
        <Card icon={<History size={32} />} title="Archives" desc="Historique OCEAN" onClick={() => navigate("/history")} />
        <Card icon={<BarChart3 size={32} />} title="Stats" desc="Données globales" onClick={() => navigate("/stats")} />
      </div>
    </div>
  );
};

const Card = ({ icon, title, desc, onClick }) => (
  <div onClick={onClick} className="cursor-pointer bg-white rounded-[2rem] p-8 shadow-sm hover:shadow-xl transition-all border border-gray-100 hover:-translate-y-2 group">
    <div className="text-blue-600 mb-6 group-hover:scale-110 transition-transform">{icon}</div>
    <h2 className="text-xl font-black mb-2 text-gray-800">{title}</h2>
    <p className="text-gray-400 text-sm font-medium">{desc}</p>
  </div>
);

export default Dashboard;