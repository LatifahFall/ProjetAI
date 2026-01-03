import React, { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";
import { useNavigate } from "react-router-dom";
import { Calendar, User, MessageCircle, ArrowLeft, History as HistoryIcon } from "lucide-react";
import { motion } from "framer-motion";

const History = () => {
  const navigate = useNavigate();
  const agentId = localStorage.getItem("agent_id");
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!agentId) {
      navigate("/login");
    }
  }, [agentId, navigate]);

  useEffect(() => {
    const fetchAnalyses = async () => {
      if (!agentId) return;

      try {
        const { data, error } = await supabase
          .from("analyses")
          .select(`
            id,
            date_analyse,
            transcription,
            score_o,
            score_c,
            score_e,
            score_a,
            score_n,
            clients!inner (
              id,
              nom,
              agent_id
            )
          `)
          .eq("clients.agent_id", agentId)
          .order("date_analyse", { ascending: false });

        if (error) throw error;
        setAnalyses(data || []);
      } catch (error) {
        // Erreur silencieuse pour l'utilisateur, pas de log front
      } finally {
        setLoading(false);
      }
    };

    fetchAnalyses();
  }, [agentId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
          <p className="font-black text-gray-400 uppercase tracking-widest text-xs">Chargement de l'historique...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 px-6 py-12 lg:px-20">
      <div className="max-w-6xl mx-auto">
        
        {/* Bouton Retour */}
        <button
          onClick={() => navigate('/dashboard')}
          className="group flex items-center gap-2 text-gray-400 hover:text-indigo-600 font-bold mb-8 transition-colors"
        >
          <ArrowLeft size={20} className="group-hover:-translate-x-1 transition-transform" />
          Dashboard
        </button>

        {/* Header */}
        <div className="flex items-center gap-6 mb-12">
          <div className="w-16 h-16 bg-white rounded-[1.5rem] shadow-sm flex items-center justify-center text-indigo-600">
            <HistoryIcon size={32} />
          </div>
          <div>
            <h1 className="text-4xl font-black text-gray-900">Historique</h1>
            <p className="text-gray-500 font-medium">Retrouvez toutes vos analyses passées</p>
          </div>
        </div>

        {analyses.length === 0 ? (
          <div className="bg-white rounded-[3rem] p-20 text-center border border-dashed border-gray-200">
            <HistoryIcon className="mx-auto text-gray-200 mb-6" size={64} />
            <p className="text-gray-400 font-bold text-xl">Aucune analyse enregistrée pour le moment</p>
            <button 
              onClick={() => navigate('/capture')}
              className="mt-6 text-indigo-600 font-black uppercase tracking-widest text-xs hover:underline"
            >
              Lancer une première analyse
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {analyses.map((a, index) => (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                key={a.id}
                className="bg-white rounded-[2.5rem] p-8 shadow-sm border border-gray-50 hover:shadow-md transition-shadow"
              >
                {/* Header Analyse */}
                <div className="flex flex-wrap justify-between items-center mb-6 gap-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                      <User size={20} />
                    </div>
                    <div>
                      <p className="font-black text-gray-900 text-lg">
                        {a.clients?.nom || "Client inconnu"}
                      </p>
                      <div className="flex items-center gap-2 text-xs text-gray-400 font-bold uppercase tracking-wider">
                        <Calendar size={14} />
                        {new Date(a.date_analyse).toLocaleDateString('fr-FR', {
                          day: 'numeric', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit'
                        })}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Transcription avec style "bulle" */}
                <div className="bg-gray-50 rounded-[2rem] p-6 mb-8 border border-gray-100 relative">
                  <div className="absolute -top-3 left-6 bg-white px-3 py-1 rounded-full border border-gray-100 flex items-center gap-2 text-[10px] font-black text-gray-400 uppercase tracking-widest">
                    <MessageCircle size={10} /> Transcription
                  </div>
                  <p className="text-gray-600 italic leading-relaxed">
                    "{a.transcription || "Aucune transcription générée."}"
                  </p>
                </div>

                {/* Grille des Scores OCEAN */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <ScoreBox label="Openness" short="O" value={a.score_o} color="bg-blue-500" />
                  <ScoreBox label="Conscientious" short="C" value={a.score_c} color="bg-green-500" />
                  <ScoreBox label="Extraversion" short="E" value={a.score_e} color="bg-purple-500" />
                  <ScoreBox label="Agreeableness" short="A" value={a.score_a} color="bg-orange-500" />
                  <ScoreBox label="Neuroticism" short="N" value={a.score_n} color="bg-red-500" />
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/* Composant interne pour l'affichage des scores individuels */
const ScoreBox = ({ label, short, value, color }) => {
  const v = Number(value ?? 0).toFixed(2);
  
  return (
    <div className="bg-white rounded-2xl p-4 border border-gray-100 shadow-sm flex flex-col items-center justify-center group hover:border-indigo-200 transition-colors">
      <div className={`w-8 h-8 ${color} rounded-lg flex items-center justify-center text-white font-black text-xs mb-3 shadow-lg`}>
        {short}
      </div>
      <p className="text-[10px] font-black text-gray-400 uppercase tracking-tighter mb-1 text-center">
        {label}
      </p>
      <p className="text-2xl font-black text-gray-900">
        {v}
      </p>
    </div>
  );
};

export default History;