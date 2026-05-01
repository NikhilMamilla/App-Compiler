import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

const EXAMPLE_PROMPTS = [
  "Build a CRM with login, contacts, dashboard, role-based access, and premium plan with payments. Admins can see analytics.",
  "Build an e-commerce store with product listings, shopping cart, checkout with Stripe, order history, and an admin panel.",
  "Create a project management tool like Trello with boards, lists, cards, team collaboration, and free/pro plans.",
  "Build a healthcare patient portal where patients book appointments, view lab results, and message doctors.",
];

const STAGES = [
  { id: 1, label: "Intent Extraction",  desc: "Parsing entities, roles, features" },
  { id: 2, label: "System Design",      desc: "Architecture, pages, API groups" },
  { id: 3, label: "Schema Generation",  desc: "UI + API + DB in parallel" },
  { id: 4, label: "Refinement",         desc: "Cross-validation & repair" },
];

// ── Types ────────────────────────────────────────────────────────────────────

interface PipelineMetrics {
  final_score: number;
  initial_score: number;
  issues_found: number;
  issues_resolved: number;
  repair_calls: number;
  ready_for_codegen: boolean;
  timings: {
    stage1_ms: number;
    stage2_ms: number;
    stage3_ms: number;
    stage4_ms: number;
    total_ms: number;
    [key: string]: number;
  };
}

interface GenerationResult {
  app_name: string;
  metrics: PipelineMetrics;
  assumptions: string[];
  warnings: string[];
  schemas: {
    ui: {
      pages: Array<{
        name: string;
        route: string;
        requires_auth: boolean;
        allowed_roles: string[];
        components: Array<{ name: string; type: string }>;
      }>;
    };
    api: {
      endpoints: Array<{
        method: string;
        path: string;
        summary: string;
        roles_allowed: string[];
      }>;
    };
    db: {
      tables: Array<{
        name: string;
        columns: Array<{
          name: string;
          type: string;
          primary_key: boolean;
          foreign_key: string;
          unique: boolean;
        }>;
      }>;
      migration_order: string[];
    };
  };
}

// ── Tiny helpers ─────────────────────────────────────────────────────────────

interface BadgeProps {
  color: "green" | "red" | "yellow" | "blue" | "gray";
  children: React.ReactNode;
}

function Badge({ color, children }: BadgeProps) {
  const colors = {
    green:  "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20 dark:border-emerald-500/30",
    red:    "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20 dark:border-red-500/30",
    yellow: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20 dark:border-amber-500/30",
    blue:   "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20 dark:border-blue-500/30",
    gray:   "bg-slate-500/10 text-slate-500 dark:text-white/30 border-slate-500/20 dark:border-white/10",
  };
  return (
    <span className={`text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded border ${colors[color]}`}>
      {children}
    </span>
  );
}

function ScoreRing({ score }: { score: number }) {
  const pct   = Math.round(score);
  const color = pct >= 90 ? "#10b981" : pct >= 70 ? "#f59e0b" : "#ef4444";
  const r     = 28;
  const circ  = 2 * Math.PI * r;
  const dash  = (pct / 100) * circ;

  return (
    <div className="relative w-20 h-20 flex items-center justify-center">
      <svg className="absolute inset-0 -rotate-90" viewBox="0 0 64 64">
        <circle cx="32" cy="32" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="5" />
        <circle
          cx="32" cy="32" r={r} fill="none"
          stroke={color} strokeWidth="5"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 1s ease" }}
        />
      </svg>
      <span className="text-lg font-bold font-mono" style={{ color }}>{pct}%</span>
    </div>
  );
}

function JsonTree({ data, depth = 0 }: { data: any; depth?: number }) {
  const [collapsed, setCollapsed] = useState(depth > 1);

  if (typeof data !== "object" || data === null) {
    if (typeof data === "string")  return <span className="text-amber-300">"{data}"</span>;
    if (typeof data === "number")  return <span className="text-blue-300">{data}</span>;
    if (typeof data === "boolean") return <span className="text-purple-300">{String(data)}</span>;
    return <span className="text-white/40">null</span>;
  }

  const isArr    = Array.isArray(data);
  const keys     = Object.keys(data);
  const open     = isArr ? "[" : "{";
  const close    = isArr ? "]" : "}";

  if (keys.length === 0) return <span className="text-white/40">{open}{close}</span>;

  return (
    <span>
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="text-white/30 hover:text-white/60 transition-colors text-xs mr-1"
      >
        {collapsed ? "▶" : "▼"}
      </button>
      <span className="text-white/40">{open}</span>
      {collapsed ? (
        <button
          onClick={() => setCollapsed(false)}
          className="text-white/30 hover:text-white/50 text-xs mx-1"
        >
          {keys.length} {isArr ? "items" : "keys"} …
        </button>
      ) : (
        <div className="ml-4 border-l border-white/5 pl-3">
          {keys.map((k, i) => (
            <div key={k} className="my-0.5">
              {!isArr && (
                <span className="text-emerald-400/80 text-xs font-mono">"{k}"</span>
              )}
              {!isArr && <span className="text-white/20 text-xs"> : </span>}
              <JsonTree data={data[k]} depth={depth + 1} />
              {i < keys.length - 1 && <span className="text-white/20 text-xs">,</span>}
            </div>
          ))}
        </div>
      )}
      <span className="text-white/40">{close}</span>
    </span>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [prompt,      setPrompt]      = useState("");
  const [loading,     setLoading]     = useState(false);
  const [activeStage, setActiveStage] = useState(0);
  const [result,      setResult]      = useState<GenerationResult | null>(null);
  const [error,       setError]       = useState<string | null>(null);
  const [activeTab,   setActiveTab]   = useState("ui");
  const [elapsed,     setElapsed]     = useState(0);
  const [theme,       setTheme]       = useState<"dark" | "light">("dark");
  
  const timerRef = useRef<any>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Apply theme to body
  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === "dark") {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [theme]);

  // Tick timer while loading
  useEffect(() => {
    if (loading) {
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed(e => e + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  // Simulate stage progress while loading
  useEffect(() => {
    if (!loading) { setActiveStage(0); return; }
    const delays = [0, 8000, 18000, 30000];
    const timers = delays.map((d, i) =>
      setTimeout(() => setActiveStage(i + 1), d)
    );
    return () => timers.forEach(clearTimeout);
  }, [loading]);

  async function handleGenerate() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Pipeline failed");
      setResult(data);
      setTimeout(() => outputRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
      setActiveStage(0);
    }
  }

  const schemas = result?.schemas;

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-[#050505] text-slate-900 dark:text-white font-mono transition-colors duration-300">

      {/* ── Professional Background Grid ── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 opacity-[0.03] dark:opacity-[0.02]"
          style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")" }}
        />
        <div className="absolute inset-0 opacity-[0.1] dark:opacity-[0.08]"
          style={{ 
            backgroundImage: "linear-gradient(to right, currentColor 1px, transparent 1px), linear-gradient(to bottom, currentColor 1px, transparent 1px)", 
            backgroundSize: "64px 64px",
            maskImage: "radial-gradient(circle at center, black, transparent 95%)" 
          }}
        />
      </div>

      <div className="relative max-w-5xl mx-auto px-6 py-12 md:py-20">

        {/* ── Top Navigation / Theme Toggle ── */}
        <nav className="flex justify-between items-center mb-16">
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] animate-pulse" />
            <span className="text-slate-400 dark:text-white/30 text-[10px] tracking-[0.3em] uppercase font-bold">System v1.2 Online</span>
          </div>
          
          <button 
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="p-2 rounded-full bg-slate-200 dark:bg-white/5 border border-slate-300 dark:border-white/10 hover:border-emerald-500/50 transition-all group"
          >
            {theme === "dark" ? (
              <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 3v1m0 16v1m9-9h-1M4 9h-1m15.364-6.364l-.707.707M6.343 17.657l-.707.707m12.728 0l-.707-.707M6.343 6.343l-.707-.707M12 5a7 7 0 100 14 7 7 0 000-14z" /></svg>
            ) : (
              <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" /></svg>
            )}
          </button>
        </nav>

        {/* ── Header ── */}
        <header className="mb-14">
          <h1 className="text-5xl font-extrabold tracking-tight mb-4 bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-white/40 bg-clip-text text-transparent">
            App<span className="text-emerald-500 dark:text-emerald-400">Compiler</span>
          </h1>
          <p className="text-slate-500 dark:text-white/30 text-sm leading-relaxed max-w-xl">
            High-fidelity code generation via multi-stage reasoning.<br />
            Describe your vision, and we'll generate the blueprint.
          </p>
        </header>

        {/* ── Input Section ── */}
        <section className="mb-10">
          <div className="border border-slate-200 dark:border-white/5 rounded-2xl bg-slate-100 dark:bg-black/40 shadow-sm dark:shadow-none overflow-hidden transition-all">

            {/* Example prompts */}
            <div className="px-4 pt-4 pb-3 border-b border-slate-200 dark:border-white/5 flex gap-2 flex-wrap">
              {EXAMPLE_PROMPTS.map((p, i) => (
                <button
                  key={i}
                  onClick={() => setPrompt(p)}
                  className="text-[10px] uppercase font-bold text-slate-400 dark:text-white/30 hover:text-emerald-500 dark:hover:text-emerald-400 border border-slate-200 dark:border-white/10 hover:border-emerald-500/30 rounded px-2.5 py-1.5 transition-all"
                >
                  Example {i + 1}
                </button>
              ))}
            </div>

            <textarea
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe the app you want to build…"
              rows={4}
              className="w-full bg-transparent px-6 py-5 text-sm text-slate-700 dark:text-white/80 placeholder-slate-300 dark:placeholder-white/10 resize-none outline-none leading-relaxed"
              onKeyDown={e => { if (e.key === "Enter" && e.metaKey) handleGenerate(); }}
            />

            <div className="px-6 pb-5 flex items-center justify-between">
              <span className="text-slate-300 dark:text-white/10 text-[10px] font-bold uppercase tracking-widest">{prompt.length} chars · ⌘↵ to run</span>
              <button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className="px-6 py-2.5 bg-slate-900 dark:bg-emerald-500 hover:bg-emerald-600 dark:hover:bg-emerald-400 disabled:opacity-20 disabled:cursor-not-allowed text-white dark:text-black text-xs font-black uppercase tracking-widest rounded-xl transition-all shadow-lg dark:shadow-none"
              >
                {loading ? `Working… ${elapsed}s` : "Execute Pipeline"}
              </button>
            </div>
          </div>
        </section>

        {/* ── Pipeline Stage Progress ── */}
        {loading && (
          <section className="mb-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {STAGES.map((s) => {
                const done    = activeStage > s.id;
                const current = activeStage === s.id;
                return (
                  <div key={s.id}
                    className={`border rounded-xl p-4 transition-all duration-500 ${
                      done    ? "border-emerald-500/30 bg-emerald-500/[0.03] dark:bg-emerald-500/5" :
                      current ? "border-slate-900 dark:border-white/20 bg-slate-100 dark:bg-white/5" :
                                "border-slate-200 dark:border-white/5 bg-white dark:bg-white/[0.01] opacity-40"
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${done ? "bg-emerald-500" : current ? "bg-slate-900 dark:bg-white animate-pulse" : "bg-slate-300 dark:bg-white/10"}`} />
                      <span className={`text-[10px] font-black uppercase tracking-widest ${done ? "text-emerald-600 dark:text-emerald-400" : current ? "text-slate-900 dark:text-white" : "text-slate-400 dark:text-white/30"}`}>
                        Stage {s.id}
                      </span>
                    </div>
                    <div className="text-[11px] font-bold text-slate-500 dark:text-white/40">{s.label}</div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ── Error ── */}
        {error && (
          <div className="mb-10 border border-red-200 dark:border-red-500/30 bg-red-50 dark:bg-red-500/5 rounded-2xl px-6 py-4 text-red-600 dark:text-red-300 text-sm flex items-center gap-3">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
            <span className="font-bold">Error:</span> {error}
          </div>
        )}

        {/* ── Result ── */}
        {result && (
          <section ref={outputRef} className="animate-in fade-in slide-in-from-bottom-4 duration-700">

            {/* Metrics bar */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              {[
                { label: "App Name",     value: result.app_name },
                { label: "Final Score",  value: `${result.metrics.final_score}%` },
                { label: "Latency",      value: `${(result.metrics.timings.total_ms/1000).toFixed(1)}s` },
                { label: "Repairs",      value: `${result.metrics.issues_resolved}/${result.metrics.issues_found}` },
              ].map((m) => (
                <div key={m.label} className="border border-slate-200 dark:border-white/5 rounded-2xl px-5 py-4 bg-slate-100 dark:bg-black/40 shadow-sm dark:shadow-none">
                  <div className="text-slate-400 dark:text-white/20 text-[10px] font-black uppercase tracking-widest mb-1.5">{m.label}</div>
                  <div className="text-slate-900 dark:text-white font-black text-sm">{m.value}</div>
                </div>
              ))}
            </div>

            {/* Score + stage timings */}
            <div className="border border-slate-200 dark:border-white/5 rounded-2xl bg-slate-100 dark:bg-black/40 p-6 mb-8 flex flex-wrap gap-8 items-center shadow-sm dark:shadow-none">
              <div className="flex items-center gap-6">
                <ScoreRing score={result.metrics.final_score} />
                <div>
                  <div className="text-slate-400 dark:text-white/30 text-[10px] font-black uppercase tracking-widest mb-2">Overall Quality</div>
                  <div className="flex gap-2.5">
                    <Badge color={result.metrics.ready_for_codegen ? "green" : "red"}>
                      {result.metrics.ready_for_codegen ? "READY" : "FAILED"}
                    </Badge>
                    <Badge color="blue">REPAIRS: {result.metrics.repair_calls}</Badge>
                  </div>
                </div>
              </div>

              <div className="flex-1 grid grid-cols-4 gap-4 min-w-[300px]">
                {["stage1_ms","stage2_ms","stage3_ms","stage4_ms"].map((k,i) => (
                  <div key={k} className="border-l border-slate-100 dark:border-white/5 pl-4">
                    <div className="text-slate-400 dark:text-white/20 text-[10px] font-black uppercase tracking-widest mb-1">Stg {i+1}</div>
                    <div className="text-slate-900 dark:text-white text-sm font-black">
                      {(result.metrics.timings[k]/1000).toFixed(1)}s
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Schema tabs */}
            {schemas && (
              <div className="border border-slate-200 dark:border-white/5 rounded-2xl bg-slate-100 dark:bg-black/40 shadow-lg dark:shadow-none overflow-hidden">
                <div className="border-b border-slate-200 dark:border-white/5 flex bg-slate-200/50 dark:bg-transparent">
                  {[
                    { key: "ui",  label: "User Interface" },
                    { key: "api", label: "API Endpoints" },
                    { key: "db",  label: "Database" },
                    { key: "raw", label: "Source" },
                  ].map(t => (
                    <button
                      key={t.key}
                      onClick={() => setActiveTab(t.key)}
                      className={`px-8 py-4 text-[10px] font-black uppercase tracking-[0.2em] transition-all border-b-2 ${
                        activeTab === t.key
                          ? "border-emerald-500 text-emerald-600 dark:text-emerald-400 bg-white dark:bg-white/5"
                          : "border-transparent text-slate-400 dark:text-white/20 hover:text-slate-600 dark:hover:text-white/50"
                      }`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>

                <div className="p-8 max-h-[600px] overflow-y-auto text-xs leading-relaxed">
                  {activeTab === "ui" && schemas?.ui && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {schemas.ui.pages?.map((page) => (
                        <div key={page.name} className="border border-slate-100 dark:border-white/5 rounded-xl p-5 bg-slate-50/30 dark:bg-transparent">
                          <div className="flex items-center justify-between mb-4">
                            <span className="text-slate-900 dark:text-white font-black uppercase tracking-wider">{page.name}</span>
                            <Badge color="gray">{page.route}</Badge>
                          </div>
                          <div className="flex gap-2 mb-4">
                            {page.requires_auth
                              ? <Badge color="yellow">AUTH</Badge>
                              : <Badge color="green">PUBLIC</Badge>}
                            {page.allowed_roles?.map((r) => (
                              <Badge key={r} color="blue">{r}</Badge>
                            ))}
                          </div>
                          <div className="space-y-2">
                            {page.components?.map((c) => (
                              <div key={c.name} className="flex justify-between items-center text-[10px] text-slate-500 dark:text-white/40 bg-white dark:bg-white/5 border border-slate-100 dark:border-white/5 rounded-lg px-3 py-2">
                                <span className="font-bold text-slate-700 dark:text-white/60">{c.name}</span>
                                <span className="opacity-50">{c.type}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === "api" && schemas?.api && (
                    <div className="space-y-3">
                      {schemas.api.endpoints?.map((ep, i) => (
                        <div key={i} className="flex items-center gap-6 border border-slate-100 dark:border-white/5 rounded-xl px-5 py-4 hover:bg-slate-50 dark:hover:bg-white/5 transition-all">
                          <div className={`w-14 text-center py-1 rounded text-[10px] font-black tracking-tighter ${
                            ep.method === "GET" ? "bg-blue-500/10 text-blue-500" :
                            ep.method === "POST" ? "bg-emerald-500/10 text-emerald-500" :
                            ep.method === "DELETE" ? "bg-red-500/10 text-red-500" : "bg-amber-500/10 text-amber-500"
                          }`}>{ep.method}</div>
                          <div className="flex-1 min-w-0">
                            <div className="text-slate-800 dark:text-white/90 font-black text-sm tracking-tight">{ep.path}</div>
                            <div className="text-slate-400 dark:text-white/30 text-[10px] uppercase font-bold mt-1">{ep.summary}</div>
                          </div>
                          <div className="flex gap-1.5">
                            {ep.roles_allowed?.map((r) => (
                              <Badge key={r} color="gray">{r}</Badge>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {activeTab === "db" && schemas?.db && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {schemas.db.tables?.map((table) => (
                        <div key={table.name} className="border border-slate-100 dark:border-white/5 rounded-xl p-5 bg-slate-50/30 dark:bg-transparent">
                          <div className="flex items-center justify-between mb-4 pb-2 border-b border-slate-100 dark:border-white/5">
                            <span className="text-slate-900 dark:text-white font-black uppercase tracking-wider">{table.name}</span>
                            <span className="text-[10px] font-bold text-slate-400 dark:text-white/20">{table.columns?.length ?? 0} COLS</span>
                          </div>
                          <div className="space-y-2">
                            {table.columns?.map((col) => (
                              <div key={col.name} className="flex items-center justify-between text-[11px]">
                                <div className="flex items-center gap-2">
                                  <span className={`font-bold ${col.primary_key ? "text-emerald-600 dark:text-emerald-400" : "text-slate-700 dark:text-white/70"}`}>
                                    {col.name}
                                  </span>
                                  {col.primary_key && <span className="text-[8px] px-1 bg-emerald-500/10 text-emerald-500 rounded font-black">PK</span>}
                                  {col.foreign_key && <span className="text-[8px] px-1 bg-blue-500/10 text-blue-500 rounded font-black">FK</span>}
                                </div>
                                <span className="text-slate-400 dark:text-white/20 font-mono text-[10px] uppercase">{col.type}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                      <div className="md:col-span-2 border border-slate-100 dark:border-white/5 rounded-xl px-6 py-4 bg-slate-50/50 dark:bg-transparent">
                        <div className="text-slate-400 dark:text-white/30 text-[10px] font-black uppercase tracking-widest mb-3">Migration Dependency Order</div>
                        <div className="flex gap-3 flex-wrap">
                          {schemas.db.migration_order?.map((t, i) => (
                            <div key={t} className="flex items-center gap-2 text-xs font-bold text-slate-600 dark:text-white/60 bg-white dark:bg-white/5 border border-slate-100 dark:border-white/5 rounded-lg px-3 py-1.5">
                              <span className="text-emerald-500 dark:text-emerald-400">{i + 1}</span>
                              {t}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === "raw" && (
                    <pre className="text-white/50 text-xs leading-relaxed overflow-x-auto">
                      <JsonTree data={result} depth={0} />
                    </pre>
                  )}
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}