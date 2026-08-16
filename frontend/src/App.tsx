import { ReactElement, useState } from "react";
import { BottomNav } from "./components/BottomNav";
import { TopBar } from "./components/TopBar";
import { PageId } from "./constants";
import { AddOrder } from "./pages/AddOrder";
import { Dashboard } from "./pages/Dashboard";
import { Database } from "./pages/Database";
import { HowItWorks } from "./pages/HowItWorks";
import { NewSku } from "./pages/NewSku";
import "./App.css";

const PAGES: Record<PageId, () => ReactElement> = {
  dashboard: Dashboard,
  database: Database,
  "new-sku": NewSku,
  "add-order": AddOrder,
  "how-it-works": HowItWorks,
};

function App() {
  const [page, setPage] = useState<PageId>("dashboard");
  const Page = PAGES[page];

  return (
    <div className="app-shell">
      <TopBar />
      <main className="app-content">
        <Page />
      </main>
      <BottomNav active={page} onSelect={setPage} />
    </div>
  );
}

export default App;
