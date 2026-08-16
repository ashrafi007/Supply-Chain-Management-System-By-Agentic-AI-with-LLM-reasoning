import { ReactElement, useEffect, useRef, useState } from "react";
import { NAV_ITEMS, PageId } from "../constants";
import { AddOrderIcon, DashboardIcon, DatabaseIcon, HowItWorksIcon, NewSkuIcon } from "../icons";

const ICONS: Record<PageId, (props: { size?: number }) => ReactElement> = {
  dashboard: DashboardIcon,
  database: DatabaseIcon,
  "new-sku": NewSkuIcon,
  "add-order": AddOrderIcon,
  "how-it-works": HowItWorksIcon,
};

export function BottomNav({
  active,
  onSelect,
}: {
  active: PageId;
  onSelect: (page: PageId) => void;
}) {
  const itemRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(null);

  // Measures the active button's real position so the pill can slide to it -- avoids
  // hard-coding pixel offsets per tab, which would silently drift the moment a label
  // changes length or a tab gets added/removed. Also re-measures on window resize --
  // without that, resizing the window without switching tabs first would leave the
  // pill sitting wherever the old (now-wrong) position was.
  useEffect(() => {
    const measure = () => {
      const el = itemRefs.current[active];
      if (el) {
        setIndicator({ left: el.offsetLeft, width: el.offsetWidth });
      }
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [active]);

  return (
    <nav className="bottom-nav-wrap">
      <div className="bottom-nav">
        {indicator && (
          <div
            className="bottom-nav__indicator"
            style={{ left: indicator.left, width: indicator.width }}
          />
        )}
        {NAV_ITEMS.map((item) => {
          const Icon = ICONS[item.id];
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              ref={(el) => {
                itemRefs.current[item.id] = el;
              }}
              className={`bottom-nav__item${isActive ? " bottom-nav__item--active" : ""}`}
              onClick={() => onSelect(item.id)}
              type="button"
            >
              <Icon size={19} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
