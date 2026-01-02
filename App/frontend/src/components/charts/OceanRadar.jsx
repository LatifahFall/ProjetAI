// src/components/charts/OceanRadar.jsx
const LABELS = [
  { key: "openness", label: "Ouverture" },
  { key: "conscientiousness", label: "Conscience" },
  { key: "extraversion", label: "Extraversion" },
  { key: "agreeableness", label: "Agréabilité" },
  { key: "neuroticism", label: "Névrosisme" },
];

function polar(cx, cy, r, angleRad) {
  return { x: cx + r * Math.cos(angleRad), y: cy + r * Math.sin(angleRad) };
}

export default function OceanRadar({ ocean }) {
  const size = 320;
  const cx = size / 2;
  const cy = size / 2;
  const R = 110;

  const points = LABELS.map((it, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / LABELS.length;
    const v = Math.max(0, Math.min(100, ocean?.[it.key] ?? 0));
    const r = (v / 100) * R;
    return polar(cx, cy, r, angle);
  });

  const polygon = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className="w-full flex justify-center">
      <svg width={size} height={size} className="max-w-full">
        {/* grid */}
        {[0.25, 0.5, 0.75, 1].map((k) => (
          <circle key={k} cx={cx} cy={cy} r={R * k} fill="none" stroke="#e5e7eb" />
        ))}

        {/* axes + labels */}
        {LABELS.map((it, i) => {
          const angle = -Math.PI / 2 + (i * 2 * Math.PI) / LABELS.length;
          const end = polar(cx, cy, R, angle);
          const label = polar(cx, cy, R + 26, angle);
          return (
            <g key={it.key}>
              <line x1={cx} y1={cy} x2={end.x} y2={end.y} stroke="#e5e7eb" />
              <text
                x={label.x}
                y={label.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize="12"
                fill="#64748b"
                style={{ fontWeight: 800 }}
              >
                {it.label}
              </text>
            </g>
          );
        })}

        {/* polygon */}
        <polygon points={polygon} fill="rgba(100,108,255,0.20)" stroke="#646cff" strokeWidth="2" />

        {/* center dot */}
        <circle cx={cx} cy={cy} r="3" fill="#646cff" />
      </svg>
    </div>
  );
}
