export function StatTile({
  label,
  value,
  sublabel,
  tone = "neutral",
}: {
  label: string;
  value: string;
  sublabel?: string;
  tone?: "good" | "neutral";
}) {
  return (
    <div className={`stat-tile${tone === "good" ? " stat-tile--good" : ""}`}>
      <div className="stat-tile__value">{value}</div>
      <div className="stat-tile__label">{label}</div>
      {sublabel && <div className="stat-tile__sublabel">{sublabel}</div>}
    </div>
  );
}
