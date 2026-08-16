// Name not decided yet (per your spec) -- placeholder, change this one line whenever
// you pick a real name. Shown top-center per the layout spec.
export const APP_NAME = "Supply Chain Copilot";

export type PageId = "dashboard" | "database" | "new-sku" | "add-order" | "how-it-works";

export const NAV_ITEMS: { id: PageId; label: string }[] = [
  { id: "dashboard", label: "Dashboard" },
  { id: "database", label: "Database" },
  { id: "new-sku", label: "New SKU" },
  { id: "add-order", label: "Add Order" },
  { id: "how-it-works", label: "How It Works" },
];
