// src/components/cards/ActionCard.jsx
import { ArrowRight } from "lucide-react";

export default function ActionCard({ icon, title, description, cta, onClick, gradient = false, disabled = false }) {
  return (
    <div
      onClick={disabled ? undefined : onClick}
      className={[
        "rounded-3xl p-8 border transition-all",
        disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer hover:shadow-lg",
        gradient
          ? "bg-gradient-to-br from-blue-50 to-white border-blue-100 hover:border-blue-200"
          : "bg-white border-gray-100 hover:border-gray-200",
      ].join(" ")}
    >
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6 bg-gray-50">
        {icon}
      </div>

      <h3 className="text-xl font-black text-slate-900 mb-2">{title}</h3>
      <p className="text-slate-500 text-sm leading-relaxed mb-4">{description}</p>

      {cta ? (
        <div className="flex items-center text-[#646cff] font-black text-sm">
          <span>{cta}</span>
          <ArrowRight className="w-4 h-4 ml-2" />
        </div>
      ) : (
        <span className="text-xs text-slate-400 font-semibold">Bientôt disponible</span>
      )}
    </div>
  );
}
