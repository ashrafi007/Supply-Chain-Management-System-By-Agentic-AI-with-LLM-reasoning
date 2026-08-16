import { GRADE_COLOR } from "./SkuResultCard";

const GRADES = ["A", "B", "C", "D"] as const;

/** Aggregate Supplier Auditor grade breakdown across every currently-queued SKU --
 * the "agent-wide" counterpart to each card's own per-SKU risk profile. */
export function GradeDistribution({ distribution }: { distribution: Record<string, number> }) {
  const total = GRADES.reduce((sum, g) => sum + (distribution[g] ?? 0), 0);

  if (total === 0) {
    return <p className="muted">No graded suppliers yet.</p>;
  }

  return (
    <div className="grade-dist">
      {GRADES.map((g) => {
        const count = distribution[g] ?? 0;
        const pct = (count / total) * 100;
        return (
          <div className="grade-dist__row" key={g}>
            <span className="grade-dist__label" style={{ color: GRADE_COLOR[g] }}>
              Grade {g}
            </span>
            <div className="grade-dist__track">
              <div className="grade-dist__fill" style={{ width: `${pct}%`, background: GRADE_COLOR[g] }} />
            </div>
            <span className="grade-dist__count">{count}</span>
          </div>
        );
      })}
    </div>
  );
}
