// src/pages/Clients.jsx
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import DashboardLayout from "../layout/DashboardLayout";
import ClientCard from "../components/cards/ClientCard";
import { addClient, getClients, getLastAnalysisDate } from "../services/storage";

function formatDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString();
}

export default function Clients() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [company, setCompany] = useState("");
  const [refresh, setRefresh] = useState(0);

  const clients = useMemo(() => getClients(), [refresh]);

  const onAdd = (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    const c = addClient({ name: name.trim(), company: company.trim() });
    setName("");
    setCompany("");
    setRefresh((x) => x + 1);
    navigate(`/clients/${c.id}`);
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6 mb-8">
        <div>
          <h1 className="text-3xl font-black text-slate-900">Clients</h1>
          <p className="text-slate-500 font-semibold text-sm mt-1">
            Ajoutez des clients et consultez leur historique d’analyses.
          </p>
        </div>

        <form onSubmit={onAdd} className="bg-white border border-gray-100 rounded-3xl p-5 flex flex-col sm:flex-row gap-3">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Nom du client"
            className="px-4 py-3 rounded-2xl border border-gray-100 bg-gray-50 outline-none focus:ring-2 focus:ring-[#646cff]"
          />
          <input
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Entreprise (optionnel)"
            className="px-4 py-3 rounded-2xl border border-gray-100 bg-gray-50 outline-none focus:ring-2 focus:ring-[#646cff]"
          />
          <button
            type="submit"
            className="bg-gradient-ocean text-white px-6 py-3 rounded-2xl font-black shadow-lg hover:scale-[1.01] transition"
          >
            + Ajouter
          </button>
        </form>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {clients.length === 0 ? (
          <div className="bg-white rounded-3xl border border-gray-100 p-8 text-slate-500 font-semibold">
            Aucun client pour le moment. Ajoute un client pour commencer.
          </div>
        ) : (
          clients.map((c) => {
            const last = getLastAnalysisDate(c.id);
            return (
              <ClientCard
                key={c.id}
                client={c}
                lastAnalysisText={last ? formatDate(last) : null}
                onClick={() => navigate(`/clients/${c.id}`)}
              />
            );
          })
        )}
      </div>
    </DashboardLayout>
  );
}
