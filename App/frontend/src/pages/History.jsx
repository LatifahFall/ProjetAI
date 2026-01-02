import { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";
import { Calendar, User, MessageCircle } from "lucide-react";

const History = () => {
  const agentId = localStorage.getItem("agent_id");
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!agentId) {
      window.location.href = "/login";
    }
  }, [agentId]);

  useEffect(() => {
    const fetchAnalyses = async () => {
      if (!agentId) return;

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

      if (error) {
        console.error("Erreur Supabase :", error.message);
      } else {
        setAnalyses(data || []);
      }

      setLoading(false);
    };

    fetchAnalyses();
  }, [agentId]);

  if (loading) {
    return (
      <div className="p-10 text-slate-400 font-bold">
        Chargement de l’historique…
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-10">
      <h1 className="text-4xl font-black mb-10 text-slate-900">
        Historique des analyses
      </h1>

      {analyses.length === 0 ? (
        <p className="text-slate-400 font-bold">
          Aucune analyse trouvée.
        </p>
      ) : (
        <div className="space-y-8 max-w-5xl">
          {analyses.map((a) => (
            <div
              key={a.id}
              className="bg-white/80 backdrop-blur-xl rounded-[2.5rem] p-8 shadow-xl border border-white"
            >
              {/* Header */}
              <div className="flex flex-wrap justify-between items-center mb-6 gap-4">
                <div className="flex items-center gap-3">
                  <div className="bg-indigo-600 text-white p-2 rounded-xl">
                    <User size={16} />
                  </div>
                  <div>
                    <p className="font-black text-slate-900">
                      {a.clients?.nom || "Client inconnu"}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-slate-400 font-bold">
                      <Calendar size={12} />
                      {new Date(a.date_analyse).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Transcription */}
              <div className="bg-slate-50 rounded-2xl p-6 mb-6 border border-slate-100">
                <div className="flex items-center gap-2 mb-2 text-slate-500 font-black text-xs uppercase">
                  <MessageCircle size={14} />
                  Transcription
                </div>
                <p className="text-slate-700 italic">
                  “{a.transcription || "Aucune transcription"}”
                </p>
              </div>

              {/* Scores */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <Score label="Openness" short="O" value={a.score_o} />
                <Score label="Conscientiousness" short="C" value={a.score_c} />
                <Score label="Extraversion" short="E" value={a.score_e} />
                <Score label="Agreeableness" short="A" value={a.score_a} />
                <Score label="Neuroticism" short="N" value={a.score_n} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const Score = ({ label, short, value }) => {
  const v = Number(value ?? 0).toFixed(2);

  return (
    <div className="bg-slate-50 rounded-2xl p-5 text-center shadow-inner">
      <p className="text-xs font-black text-slate-400 uppercase">
        {label}
      </p>
      <p className="text-3xl font-black text-indigo-600 mt-2">
        {v}
      </p>
      <p className="text-xs font-bold text-slate-400 mt-1">
        {short}
      </p>
    </div>
  );
};

export default History;
