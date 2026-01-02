// src/pages/NewRecording.jsx
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import DashboardLayout from "../layout/DashboardLayout";
import AudioRecorder from "../components/audio/AudioRecorder";
import { getClientById, addAnalysis } from "../services/storage";
import { predictFromAudioBlob } from "../services/api";

export default function NewRecording() {
  const { id } = useParams();
  const navigate = useNavigate();
  const client = useMemo(() => getClientById(id), [id]);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  if (!client) {
    return (
      <DashboardLayout>
        <div className="bg-white p-8 rounded-3xl border border-gray-100">
          Client introuvable. <Link className="text-[#646cff] font-black" to="/clients">Retour</Link>
        </div>
      </DashboardLayout>
    );
  }

  const onRecorded = async (blob, url) => {
    setLoading(true);
    setResult(null);

    try {
      const pred = await predictFromAudioBlob(blob);

      const analysis = {
        id: crypto.randomUUID(),
        clientId: id,
        createdAt: new Date().toISOString(),
        audioUrl: url, // pour l’instant local object url
        score: pred.score,
        ocean: pred.ocean,
        advice: pred.advice,
      };

      addAnalysis(analysis);
      setResult(analysis);

      // redirect auto après 1s
      setTimeout(() => navigate(`/clients/${id}`), 900);
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="mb-8">
        <Link to={`/clients/${id}`} className="text-sm font-black text-slate-500 hover:text-[#646cff]">
          ← Retour Historique
        </Link>
        <h1 className="text-3xl font-black text-slate-900 mt-2">
          Nouvelle analyse – {client.name}
        </h1>
        <p className="text-slate-500 font-semibold text-sm">
          Enregistrez un audio puis lancement automatique de la prédiction.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <AudioRecorder onRecorded={onRecorded} />

        <div className="bg-white rounded-3xl border border-gray-100 p-8">
          <h2 className="font-black text-lg text-slate-900 mb-2">🧠 Analyse IA</h2>
          <p className="text-sm text-slate-500 mb-6">Résultat après arrêt de l’enregistrement.</p>

          {loading && (
            <div className="flex items-center gap-3 text-slate-600 font-semibold">
              <div className="w-6 h-6 border-2 border-[#646cff] border-t-transparent rounded-full animate-spin" />
              Analyse en cours…
            </div>
          )}

          {!loading && !result && (
            <div className="text-slate-400 font-semibold text-sm">
              Aucun résultat pour le moment.
            </div>
          )}

          {result && (
            <div className="mt-4">
              <p className="text-xs text-slate-400 font-bold">Score global</p>
              <p className="text-4xl font-black text-[#646cff]">{result.score}%</p>

              <div className="mt-6 grid grid-cols-2 gap-3">
                {Object.entries(result.ocean).map(([k, v]) => (
                  <div key={k} className="border border-gray-100 rounded-2xl p-4">
                    <p className="text-xs uppercase font-black text-slate-400">{k}</p>
                    <p className="text-xl font-black text-slate-900 mt-2">{v}%</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
