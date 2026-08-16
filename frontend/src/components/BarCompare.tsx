export interface BarCompareItem {
  label: string;
  value: number;
  max: number;
  displayValue: string;
  highlight?: boolean;
}

/** Single-axis horizontal bar comparison -- "ours vs. baseline" stories. Per the
 * dataviz skill: one axis, thin marks, direct labels, the emphasized entry in the
 * accent hue and everything else in de-emphasis gray (not a full categorical palette
 * for what's really a 2-3 item comparison). */
export function BarCompare({ items }: { items: BarCompareItem[] }) {
  return (
    <div className="bar-compare">
      {items.map((item) => (
        <div className="bar-compare__row" key={item.label}>
          <span className="bar-compare__label">{item.label}</span>
          <div className="bar-compare__track">
            <div
              className={`bar-compare__fill${item.highlight ? " bar-compare__fill--accent" : ""}`}
              style={{ width: `${Math.max(2, Math.min(100, (item.value / item.max) * 100))}%` }}
            />
          </div>
          <span className={`bar-compare__value${item.highlight ? " bar-compare__value--accent" : ""}`}>
            {item.displayValue}
          </span>
        </div>
      ))}
    </div>
  );
}
