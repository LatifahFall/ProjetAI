// src/pages/ClientDetails.jsx
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import DashboardLayout from "../layout/DashboardLayout";
import OceanRadar from "../components/charts/OceanRadar";
import { getClientById, getAnalysesByClient } from "../services/storage";

function formatDateTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString();
}

export default function ClientDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  const client = useMemo(() => getClientById(id), [id]);
  const analyses = useMemo(() => getAnalysesByClient(id), [id]);

  const [selected, setSelected] = useState(analyses[0] || null);

  if (!client) {
    return (
      <DashboardLayout>
        <div className="bg-white p-8 rounded-3xl border border-gray-100">
          Client introuvable. <Link className="text-[#646cff] font-black" to="/clients">Retour</Link>
        </div>
      </DashboardLayout>
    );
  }

  const startNew = () => navigate(`/clients/${id}/new`);

  return (
    <DashboardLayout>
      <div className="flex items-start justify-between gap-6 mb-8">
        <div>
          <Link to="/clients" className="text-sm font-black text-slate-500 hover:text-[#646cff]">
            ← Retour Clients
          </Link>
          <h1 className="text-3xl font-black text-slate-900 mt-2">{client.name}</h1>
          <p className="text-slate-500 font-semibold text-sm">{client.company || "—"}</p>
        </div>

        <button
          onClick={startNew}
          className="bg-gradient-ocean text-white px-6 py-3 rounded-2xl font-black shadow-lg hover:scale-[1.01] transition"
        >
          + Nouvelle analyse
        </button>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Historique */}
        <div className="lg:col-span-1 bg-white rounded-3xl border border-gray-100 p-6">
          <h2 className="font-black text-slate-900 mb-4">Historique</h2>

          {analyses.length === 0 ? (
            <div className="text-sm text-slate-500 font-semibold">
              Aucune analyse. Lance une nouvelle analyse.
            </div>
          ) : (
            <div className="space-y-3">
              {analyses.map((a, idx) => (
                <button
                  key={a.id}
                  onClick={() => setSelected(a)}
                  className={[
                    "w-full text-left rounded-2xl border p-4 transition",
                    selected?.id === a.id
                      ? "border-blue-200 bg-blue-50/40"
                      : "border-gray-100 hover:border-gray-200 hover:bg-gray-50",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-black text-slate-900">Analyse #{analyses.length - idx}</p>
                    <p className="font-black text-[#646cff]">{a.score}%</p>
                  </div>
                  <p className="text-xs text-slate-400 font-semibold mt-1">{formatDateTime(a.createdAt)}</p>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Détails */}
        <div className="lg:col-span-2">
          {!selected ? (
            <div className="bg-white rounded-3xl border border-gray-100 p-8 text-slate-500 font-semibold">
              Sélectionne une analyse dans l’historique.
            </div>
          ) : (
            <div className="bg-white rounded-3xl border border-gray-100 p-8">
              <div className="flex items-start justify-between gap-4 mb-6">
                <div>
                  <h2 className="font-black text-slate-900 text-xl">Résultat OCEAN</h2>
                  <p className="text-sm text-slate-500 font-semibold">{formatDateTime(selected.createdAt)}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-slate-400 font-bold">Score global</p>
                  <p className="text-3xl font-black text-[#646cff]">{selected.score}%</p>
                </div>
              </div>

              <OceanRadar ocean={selected.ocean} />

              <div className="grid md:grid-cols-2 gap-4 mt-8">
                {Object.entries(selected.ocean).map(([k, v]) => (
                  <div key={k} className="border border-gray-100 rounded-2xl p-5">
                    <p className="text-xs font-black text-slate-400 uppercase tracking-widest">
                      {k}
                    </p>
                    <div className="mt-3 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-ocean" style={{ width: `${v}%` }} />
                    </div>
                    <p className="mt-3 text-2xl font-black text-slate-900">{v}%</p>
                  </div>
                ))}
              </div>

              {selected.audioUrl && (
                <div className="mt-8">
                  <p className="text-sm font-black text-slate-900 mb-2">Audio</p>
                  <audio controls src={selected.audioUrl} className="w-full" />
                </div>
              )}

              {selected.advice && (
                <div className="mt-8 bg-blue-50 border border-blue-100 rounded-2xl p-5">
                  <p className="font-black text-slate-900">Conseil IA</p>
                  <p className="text-sm text-slate-600 font-semibold mt-1">{selected.advice}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
