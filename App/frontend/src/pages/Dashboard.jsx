// src/pages/Dashboard.jsx
import { useNavigate } from "react-router-dom";
import ActionCard from "../components/cards/ActionCard";
import { Play, History, BarChart3, Sparkles, Users } from "lucide-react";
import DashboardLayout from "../layout/DashboardLayout";

export default function Dashboard() {
  const navigate = useNavigate();
  const userName = localStorage.getItem("agent_name") || "agent";

  return (
    <DashboardLayout>
      <div className="text-center mb-12">
        <h1 className="text-4xl font-black text-slate-900 mb-3">
          Bienvenue,{" "}
          <span className="bg-gradient-ocean bg-clip-text text-transparent">
            {userName}
          </span>
        </h1>
        <p className="text-slate-500 font-semibold">
          Gérez vos clients, enregistrez des audios et suivez l’évolution
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        <ActionCard
          icon={<Play className="w-7 h-7 text-[#646cff]" />}
          title="Nouvelle analyse"
          description="Sélectionnez un client et lancez une nouvelle prédiction vocale."
          cta="Démarrer"
          gradient
          onClick={() => navigate("/clients")}
        />

        <ActionCard
          icon={<Users className="w-7 h-7 text-slate-400" />}
          title="Clients"
          description="Ajoutez des clients et consultez leur historique."
          cta="Voir"
          onClick={() => navigate("/clients")}
        />

        <ActionCard
          icon={<BarChart3 className="w-7 h-7 text-slate-400" />}
          title="Statistiques"
          description="Analyses globales et tendances (à venir)."
          disabled
        />
      </div>

      <div className="bg-white rounded-3xl p-8 border border-gray-100">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-blue-50 rounded-2xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-[#646cff]" />
          </div>
          <div>
            <h3 className="text-lg font-black text-slate-900 mb-2">
              À propos
            </h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              Chaque client possède un historique d’analyses vocales. À chaque
              nouvel enregistrement, une nouvelle prédiction OCEAN est ajoutée,
              permettant le suivi dans le temps.
            </p>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
