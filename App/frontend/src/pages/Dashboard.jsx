import { useEffect } from "react";
import { Mic, History, BarChart3, Users } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Dashboard = () => {
  const navigate = useNavigate();

  useEffect(() => {
    if (!localStorage.getItem("agent_id")) {
        navigate("/login");
    }
    }, []);


  return (
    <div className="min-h-screen bg-gray-50 px-10 py-20">
      <h1 className="text-4xl font-black mb-12 text-gray-900">
        Dashboard Agent
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <Card
          icon={<Users size={36} />}
          title="Gestion Clients"
          desc="Créer et gérer vos clients"
          onClick={() => navigate("/clients")}
        />

        <Card
          icon={<Mic size={36} />}
          title="Enregistrer un audio"
          desc="Capturer et analyser la voix d'un client"
          onClick={() => navigate("/capture")}
        />

        <Card
          icon={<History size={36} />}
          title="Historique"
          desc="Consulter les analyses précédentes"
          onClick={() => navigate("/history")}
        />

        <Card
          icon={<BarChart3 size={36} />}
          title="Statistiques"
          desc="Visualiser les tendances OCEAN"
          onClick={() => navigate("/stats")}
        />
      </div>
    </div>
  );
};

const Card = ({ icon, title, desc, onClick }) => (
  <div
    onClick={onClick}
    className="cursor-pointer bg-white rounded-3xl p-8 shadow-lg hover:shadow-2xl transition-all border border-gray-100 hover:-translate-y-1"
  >
    <div className="text-blue-600 mb-6">{icon}</div>
    <h2 className="text-xl font-black mb-2">{title}</h2>
    <p className="text-gray-500 text-sm">{desc}</p>
  </div>
);

export default Dashboard;
