// Hand-rolled SVG radar/spider chart -- no charting library dependency. Deliberately
// scoped to a single series: dataviz skill's series-count ladder says 1-3 series is
// comfortable with color alone + direct labels; radar charts specifically get harder
// to read correctly past one series (overlapping polygons, area distortion), so this
// component doesn't attempt multi-series -- render two RadarCharts side by side instead
// of overlaying two polygons on one.
//
// Everything -- rings, polygon, axis labels, value labels -- is guaranteed to stay
// within the declared `size x size` box. That's load-bearing, not cosmetic: this
// component gets embedded in CSS grid/flex layouts alongside sibling content (e.g.
// SkuResultCard's radar-next-to-text layout). A label allowed to overflow past `size`
// spills onto whatever sits next to or outside it, since the grid track is sized to
// the declared box regardless. The radius is derived FROM size and a FIXED label
// width (not scaled to size -- text doesn't need more width just because the canvas
// is bigger), so a caller gets a big, legible polygon at a reasonable size instead of
// the radius shrinking to satisfy an over-cautious margin.

export interface RadarAxis {
  label: string;
  value: number; // 0..max
  max: number;
  displayValue: string; // pre-formatted, e.g. "99.4%" or "0.85"
}

const LABEL_FRACTION = 1.2; // axis-label distance from center, as a multiple of radius
const LABEL_WIDTH = 68; // fixed -- text width doesn't need to scale with canvas size
const VALUE_OFFSET = 0.17; // how far outside the ACTUAL data point the value label sits
// Fixed absolute cap, NOT relative to LABEL_FRACTION: a value near its max (e.g. a
// D-grade supplier at 0.85, or a 1.00 PR-AUC) would otherwise have its label capped
// back down almost exactly onto its own dot -- a relative cap shrinks toward zero
// exactly when the real value is high, which is the opposite of what's needed.
const VALUE_FRACTION_CAP = 1.08;

export function RadarChart({
  axes,
  size = 320,
  color = "var(--accent)",
}: {
  axes: RadarAxis[];
  size?: number;
  color?: string;
}) {
  const center = size / 2;
  // Largest radius for which an axis label (anchored at LABEL_FRACTION*radius, growing
  // LABEL_WIDTH further outward) still ends exactly at the box edge, never past it.
  const maxRadiusForLabels = (size / 2 - LABEL_WIDTH) / LABEL_FRACTION;
  const radius = Math.max(24, Math.min(size / 2 - 16, maxRadiusForLabels));
  const ringCount = 4;
  const n = axes.length;

  const angleFor = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;

  const pointFor = (i: number, fraction: number) => {
    const angle = angleFor(i);
    return {
      x: center + Math.cos(angle) * radius * fraction,
      y: center + Math.sin(angle) * radius * fraction,
    };
  };

  const dataFractions = axes.map((a) => Math.max(0, Math.min(1, a.value / a.max)));
  const dataPoints = axes.map((_, i) => pointFor(i, dataFractions[i]));
  const dataPath = dataPoints.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <div className="radar-chart-wrap" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="radar-chart" role="img" aria-label="Radar chart">
        {/* Grid rings -- recessive, gate-safe gray */}
        {Array.from({ length: ringCount }, (_, ring) => {
          const fraction = (ring + 1) / ringCount;
          const ringPoints = axes.map((_, i) => pointFor(i, fraction));
          return (
            <polygon
              key={ring}
              points={ringPoints.map((p) => `${p.x},${p.y}`).join(" ")}
              fill="none"
              style={{ stroke: "var(--card-border)" }}
              strokeWidth={1}
            />
          );
        })}

        {/* Axis spokes */}
        {axes.map((_, i) => {
          const outer = pointFor(i, 1);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={outer.x}
              y2={outer.y}
              style={{ stroke: "var(--card-border)" }}
              strokeWidth={1}
            />
          );
        })}

        {/* Data polygon -- 2px stroke, low-opacity fill per mark spec. Scales in from
            the center on mount/update -- transform-origin is set in CSS to the SVG
            center, which is stable across renders since `center` never changes for a
            given `size`. */}
        <polygon
          className="radar-chart__polygon"
          points={dataPath}
          fillOpacity={0.18}
          strokeWidth={2}
          style={{ fill: color, stroke: color, transformOrigin: `${center}px ${center}px` }}
        />

        {/* Vertex dots */}
        {dataPoints.map((p, i) => (
          <circle
            key={i}
            className="radar-chart__dot"
            cx={p.x}
            cy={p.y}
            r={4}
            strokeWidth={1.5}
            style={{ fill: color, stroke: "var(--bg-elevated)", animationDelay: `${180 + i * 45}ms` }}
          />
        ))}
      </svg>

      {/* Axis labels -- anchored so they grow AWAY from center on every axis (up for
          the top axis, down for the bottom one, sideways for the rest) -- never toward
          where the value label sits, so the two can't collide regardless of angle. */}
      {axes.map((a, i) => {
        const angle = angleFor(i);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const p = pointFor(i, LABEL_FRACTION);

        const horizontal = Math.abs(cos) >= 0.35;
        const textAlign = !horizontal ? "center" : cos > 0 ? "left" : "right";
        const translateX = textAlign === "center" ? "-50%" : textAlign === "left" ? "0%" : "-100%";

        const stronglyVertical = Math.abs(sin) >= 0.85;
        const translateY = !stronglyVertical ? "-50%" : sin < 0 ? "-100%" : "0%";

        return (
          <div
            key={i}
            className="radar-chart__label"
            style={{ left: p.x, top: p.y, width: LABEL_WIDTH, textAlign, transform: `translate(${translateX}, ${translateY})` }}
          >
            {a.label}
          </div>
        );
      })}

      {/* Value labels -- anchored to the ACTUAL data point (not a fixed ring), so they
          always sit just outside that SKU's real vertex regardless of its value, never
          drifting onto a neighboring vertex or the dot marker itself. Capped short of
          LABEL_FRACTION so it can never reach into axis-label territory either. */}
      {axes.map((a, i) => {
        const fraction = Math.min(dataFractions[i] + VALUE_OFFSET, VALUE_FRACTION_CAP);
        const p = pointFor(i, fraction);
        return (
          <div key={i} className="radar-chart__value" style={{ left: p.x, top: p.y, color }}>
            {a.displayValue}
          </div>
        );
      })}
    </div>
  );
}
