import React, { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";
import { useNavigate } from "react-router-dom";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer
} from "recharts";
import { ArrowLeft, BarChart3, Download, UserCheck } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Stats = () => {
  const navigate = useNavigate();
  const agentId = localStorage.getItem("agent_id");

  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState("");
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!agentId) {
      navigate("/login");
    }
  }, [agentId, navigate]);

  // Récupération de la liste des clients
  useEffect(() => {
    if (!agentId) return;
    const fetchClients = async () => {
      const { data } = await supabase
        .from("clients")
        .select("id, nom")
        .eq("agent_id", agentId)
        .order("nom");
      setClients(data || []);
    };
    fetchClients();
  }, [agentId]);

  // Déclenchement de la récupération des stats
  useEffect(() => {
    if (!selectedClient) {
      setStats(null);
      return;
    }
    fetchStats(selectedClient);
  }, [selectedClient]);

  const fetchStats = async (clientId) => {
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from("analyses")
        .select("score_o, score_c, score_e, score_a, score_n")
        .eq("client_id", clientId);

      if (error) throw error;

      if (!data || data.length === 0) {
        setStats(null);
      } else {
        const avg = (arr) => arr.reduce((s, v) => s + (v ?? 0), 0) / arr.length;
        
        // Formatage pour Recharts
        setStats([
          { trait: "Ouverture", value: avg(data.map(d => d.score_o)), fullMark: 1 },
          { trait: "Conscience", value: avg(data.map(d => d.score_c)), fullMark: 1 },
          { trait: "Extraversion", value: avg(data.map(d => d.score_e)), fullMark: 1 },
          { trait: "Agréabilité", value: avg(data.map(d => d.score_a)), fullMark: 1 },
          { trait: "Névrosisme", value: avg(data.map(d => d.score_n)), fullMark: 1 }
        ]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 px-6 py-12 lg:px-20">
      <div className="max-w-6xl mx-auto">
        
        {/* Navigation Retour */}
        <button
          onClick={() => navigate('/dashboard')}
          className="group flex items-center gap-2 text-gray-400 hover:text-indigo-600 font-bold mb-8 transition-colors"
        >
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          Dashboard
        </button>

        {/* Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-12">
          <div className="flex items-center gap-6">
            <div className="w-16 h-16 bg-white rounded-[1.5rem] shadow-sm flex items-center justify-center text-indigo-600">
              <BarChart3 size={32} />
            </div>
            <div>
              <h1 className="text-4xl font-black text-gray-900">Analyses OCEAN</h1>
              <p className="text-gray-500 font-medium">Visualisez le profil psychologique moyen</p>
            </div>
          </div>

          {/* Sélecteur Client Stylisé */}
          <div className="w-full md:w-72">
            <label className="block text-[10px] font-black uppercase tracking-[0.2em] text-gray-400 mb-2 ml-1">
              Sélectionner un client
            </label>
            <div className="relative">
              <select
                value={selectedClient}
                onChange={(e) => setSelectedClient(e.target.value)}
                className="w-full pl-5 pr-10 py-4 bg-white border-none rounded-2xl shadow-sm font-bold text-gray-700 appearance-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
              >
                <option value="">— Choisir —</option>
                {clients.map((c) => (
                  <option key={c.id} value={c.id}>{c.nom}</option>
                ))}
              </select>
              <UserCheck className="absolute right-4 top-1/2 -translate-y-1/2 text-indigo-500 pointer-events-none" size={20} />
            </div>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {loading ? (
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="bg-white rounded-[3rem] p-32 flex flex-col items-center justify-center border border-gray-100"
            >
              <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mb-4"></div>
              <p className="font-black text-gray-400 uppercase tracking-widest text-xs">Calcul des moyennes...</p>
            </motion.div>
          ) : !selectedClient ? (
            <motion.div 
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-[3rem] p-32 text-center border border-dashed border-gray-200"
            >
              <BarChart3 className="mx-auto text-gray-200 mb-6" size={64} />
              <p className="text-gray-400 font-bold text-xl">Sélectionnez un client pour voir ses statistiques</p>
            </motion.div>
          ) : !stats ? (
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="bg-white rounded-[3rem] p-32 text-center border border-gray-100"
            >
              <p className="text-gray-400 font-bold text-xl">Aucune donnée trouvée pour ce client.</p>
              <button onClick={() => navigate('/capture')} className="mt-4 text-indigo-600 font-black text-xs uppercase hover:underline">Lancer une analyse</button>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="grid grid-cols-1 lg:grid-cols-3 gap-8"
            >
              {/* Carte Graphique Radar */}
              <div className="lg:col-span-2 bg-white rounded-[3rem] p-10 shadow-sm border border-gray-50 flex flex-col items-center">
                <div className="w-full flex justify-between items-center mb-10">
                  <h3 className="font-black text-gray-900 uppercase tracking-widest text-sm">Profil Psychologique</h3>
                  <button className="flex items-center gap-2 text-xs font-black text-indigo-600 bg-indigo-50 px-4 py-2 rounded-full hover:bg-indigo-100 transition-colors">
                    <Download size={14} /> PDF
                  </button>
                </div>
                
                <div className="h-[400px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={stats}>
                      <PolarGrid stroke="#f1f5f9" />
                      <PolarAngleAxis dataKey="trait" tick={{ fill: '#94a3b8', fontSize: 12, fontWeight: 700 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                      <Radar
                        name="Moyenne"
                        dataKey="value"
                        stroke="#6366f1"
                        strokeWidth={3}
                        fill="#6366f1"
                        fillOpacity={0.2}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Liste des Scores */}
              <div className="flex flex-col gap-4">
                {stats.map((s, i) => (
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    key={s.trait}
                    className="bg-white rounded-3xl p-6 border border-gray-50 shadow-sm flex justify-between items-center group hover:border-indigo-100 transition-all"
                  >
                    <div>
                      <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">{s.trait}</p>
                      <div className="h-1.5 w-32 bg-gray-100 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }} 
                          animate={{ width: `${s.value * 100}%` }} 
                          transition={{ duration: 1, ease: "easeOut" }}
                          className="h-full bg-indigo-500"
                        />
                      </div>
                    </div>
                    <span className="text-3xl font-black text-gray-900 group-hover:text-indigo-600 transition-colors">
                      {Math.round(s.value * 100)}%
                    </span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Stats;