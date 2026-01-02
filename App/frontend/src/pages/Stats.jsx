import { useEffect, useState } from "react";
import { supabase } from "../lib/supabaseClient";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer
} from "recharts";

const Stats = () => {
  const agentId = localStorage.getItem("agent_id");

  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState("");
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!localStorage.getItem("agent_id")) {
        window.location.href = "/login";
    }
    }, []);

  // ─────────────────────────────────────────
  // Fetch clients
  // ─────────────────────────────────────────
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

  // ─────────────────────────────────────────
  // Fetch stats by client
  // ─────────────────────────────────────────
  useEffect(() => {
    if (!selectedClient) {
      setStats(null);
      return;
    }
    fetchStats(selectedClient);
  }, [selectedClient]);

  const fetchStats = async (clientId) => {
    setLoading(true);

    const { data } = await supabase
      .from("analyses")
      .select("score_o, score_c, score_e, score_a, score_n")
      .eq("client_id", clientId);

    if (!data || data.length === 0) {
      setStats(null);
      setLoading(false);
      return;
    }

    const avg = (arr) =>
      arr.reduce((s, v) => s + (v ?? 0), 0) / arr.length;

    setStats([
      { trait: "Openness", value: avg(data.map(d => d.score_o)) },
      { trait: "Conscientiousness", value: avg(data.map(d => d.score_c)) },
      { trait: "Extraversion", value: avg(data.map(d => d.score_e)) },
      { trait: "Agreeableness", value: avg(data.map(d => d.score_a)) },
      { trait: "Neuroticism", value: avg(data.map(d => d.score_n)) }
    ]);

    setLoading(false);
  };

  // ─────────────────────────────────────────
  // UI
  // ─────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-10">
      
      <h1 className="text-4xl font-black text-slate-900 mb-10">
        Statistiques OCEAN
      </h1>

      <div className="max-w-5xl mx-auto bg-white/80 backdrop-blur-xl rounded-[3rem] shadow-2xl p-10 border border-white">

        {/* Sélecteur client */}
        <div className="mb-10 max-w-sm">
          <label className="block text-xs font-black uppercase tracking-widest text-slate-500 mb-2">
            Client
          </label>
          <select
            value={selectedClient}
            onChange={(e) => setSelectedClient(e.target.value)}
            className="w-full px-5 py-4 rounded-2xl border border-slate-200 bg-white font-bold text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">— Choisir un client —</option>
            {clients.map((c) => (
              <option key={c.id} value={c.id}>
                {c.nom}
              </option>
            ))}
          </select>
        </div>

        {/* États */}
        {loading && (
          <p className="text-slate-400 font-bold">
            Chargement des statistiques…
          </p>
        )}

        {!loading && !stats && selectedClient && (
          <p className="text-slate-400 font-bold">
            Aucune analyse disponible pour ce client.
          </p>
        )}

        {/* Radar + scores */}
        {!loading && stats && (
          <>
            {/* Radar */}
            <div className="h-[420px] w-full mb-12">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={stats}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="trait" />
                  <PolarRadiusAxis domain={[0, 1]} />
                  <Radar
                    dataKey="value"
                    stroke="#4f46e5"
                    fill="#6366f1"
                    fillOpacity={0.5}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Scores */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
              {stats.map((s) => (
                <div
                  key={s.trait}
                  className="bg-slate-50 rounded-2xl p-6 text-center shadow-inner"
                >
                  <p className="text-xs font-black text-slate-400 uppercase">
                    {s.trait[0]}
                  </p>
                  <p className="text-2xl font-black text-indigo-600 mt-2">
                    {s.value.toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Stats;
