// src/components/cards/ClientCard.jsx
export default function ClientCard({ client, lastAnalysisText, onClick }) {
  return (
    <div
      onClick={onClick}
      className="bg-white rounded-3xl p-6 border border-gray-100 hover:shadow-md transition cursor-pointer"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="font-black text-lg text-slate-900">{client.name}</h3>
          <p className="text-sm text-slate-500">{client.company || "—"}</p>
        </div>

        <div className="text-xs font-bold text-slate-400">
          {lastAnalysisText ? `Dernière analyse : ${lastAnalysisText}` : "Aucune analyse"}
        </div>
      </div>
    </div>
  );
}
