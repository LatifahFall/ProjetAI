// src/layout/DashboardLayout.jsx
import { Brain, LogOut, User } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

export default function DashboardLayout({ children }) {
  const navigate = useNavigate();
  const userName = localStorage.getItem("agent_name") || "agent";

  const logout = () => {
    // TODO: si tu utilises supabase auth, tu signOut ici
    localStorage.removeItem("agent_id");
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50/40 to-white">
      <header className="border-b border-gray-100 bg-white/70 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/dashboard" className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-ocean rounded-xl flex items-center justify-center shadow-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-black tracking-tight text-slate-900">
              OCEAN<span className="text-[#646cff]">INSIGHT</span>
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-6 text-sm font-bold text-slate-500">
            <Link to="/clients" className="hover:text-[#646cff] transition">Clients</Link>
            <Link to="/dashboard" className="hover:text-[#646cff] transition">Dashboard</Link>
          </nav>

          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 text-sm text-slate-500 font-bold">
              <User className="w-4 h-4" />
              <span>{userName}</span>
            </div>

            <button
              onClick={logout}
              className="text-slate-500 hover:text-slate-900 font-black text-sm flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Déconnexion
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-12">{children}</main>
    </div>
  );
}
