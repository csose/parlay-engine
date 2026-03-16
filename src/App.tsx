import { useState, useRef, useCallback } from "react";

// ─── Types ──────────────────────────────────────────────────────────────────
type TierType = "LOCK" | "STRONG" | "VALUE";

interface Bet {
  player: string;
  prop: string;
  odds: number;
  game: string;
  tier: TierType;
  trueProb: number;
  ev: string;
  kelly: string;
}

interface ImageFile {
  file: File;
  url: string;
  name: string;
}

// ─── Helpers ────────────────────────────────────────────────────────────────
const fmtOdds  = (o: number) => (o > 0 ? `+${o}` : `${o}`);
const fdUrl    = (n: string) => `https://sportsbook.fanduel.com/search?q=${encodeURIComponent(n)}`;
const toBase64 = (file: File): Promise<string> =>
  new Promise((res, rej) => {
    const r = new FileReader();
    r.onload  = () => res((r.result as string).split(",")[1]);
    r.onerror = () => rej(new Error("Read failed"));
    r.readAsDataURL(file);
  });

const oddsNote = (o: number) => {
  if (o <= -300) return "Very likely — small payout";
  if (o <= -150) return "Likely — moderate payout";
  if (o <    0)  return "Slight favourite";
  if (o <  150)  return "Near even — good value";
  return "Underdog — big payout if hits";
};

const TIER_CFG   = {
  LOCK:   { color:"#00ff88", bg:"rgba(0,255,136,.14)",  border:"rgba(0,255,136,.45)",  label:"🔒 Lock"   },
  STRONG: { color:"#60c8ff", bg:"rgba(96,200,255,.13)", border:"rgba(96,200,255,.40)", label:"⚡ Strong" },
  VALUE:  { color:"#fbbf24", bg:"rgba(251,191,36,.13)", border:"rgba(251,191,36,.40)", label:"💎 Value"  },
} as const;

// ─── Main component ──────────────────────────────────────────────────────────
function ParlayEngine() {
  const [tab,       setTab]       = useState("UPLOAD");
  const [images,    setImages]    = useState<ImageFile[]>([]);
  const [bets,      setBets]      = useState<Bet[]>([]);
  const [parlay,    setParlay]    = useState<string[]>([]);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState<string | null>(null);
  const [filter,    setFilter]    = useState<"ALL" | TierType>("ALL");
  const [hovRow,    setHovRow]    = useState<number | null>(null);
  const [hovLink,   setHovLink]   = useState<number | null>(null);
  const [dragOver,  setDragOver]  = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // ── Add images ──────────────────────────────────────────────────────────────
  const addFiles = useCallback((files: FileList | null) => {
    if (!files) return;
    const imgs = Array.from(files)
      .filter(f => f.type.startsWith("image/"))
      .map(f => ({ file: f, url: URL.createObjectURL(f), name: f.name }));
    setImages(prev => [...prev, ...imgs]);
    setError(null);
  }, []);

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  // ── Run analysis ─────────────────────────────────────────────────────────────
  const analyse = async () => {
    if (!images.length) { setError("Please upload at least one screenshot first."); return; }
    setLoading(true);
    setError(null);

    try {
      const imagePayloads = [];
      for (const img of images) {
        const b64  = await toBase64(img.file);
        const mime = img.file.type || "image/png";
        imagePayloads.push({ data: b64, mediaType: mime });
      }

      const apiUrl = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/analyze-screenshots`;
      const res = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${import.meta.env.VITE_SUPABASE_ANON_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ images: imagePayloads }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `API error ${res.status}`);
      }
      const data = await res.json();

      const raw    = data.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("");
      const clean  = raw.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);

      if (!Array.isArray(parsed) || !parsed.length)
        throw new Error("No bets found in screenshots. Try clearer images.");

      // Deduplicate by player + prop
      const seen   = new Set<string>();
      const deduped = parsed.filter((b: Bet) => {
        const key = `${b.player}|${b.prop}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

      setBets(deduped);
      setParlay([]);
      setFilter("ALL");
      setTab("PICKS");
    } catch (err) {
      setError((err as Error).message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // ── Parlay helpers ───────────────────────────────────────────────────────────
  const toggleLeg = (player: string) =>
    setParlay(p => p.includes(player) ? p.filter(x => x !== player) : [...p, player]);

  const combinedOdds = () => {
    if (parlay.length < 2) return null;
    const sel     = bets.filter(b => parlay.includes(b.player));
    const dec     = sel.map(b => b.odds > 0 ? 1 + b.odds / 100 : 1 + 100 / Math.abs(b.odds));
    const combined = dec.reduce((a, c) => a * c, 1);
    const am      = combined >= 2 ? Math.round((combined - 1) * 100) : Math.round(-100 / (combined - 1));
    return {
      american: am > 0 ? `+${am}` : `${am}`,
      decimal:  combined.toFixed(2),
      win100:   Math.round((combined - 1) * 100),
    };
  };
  const co       = combinedOdds();
  const filtered = filter === "ALL" ? bets : bets.filter(b => b.tier === filter);

  // ── Shared styles ────────────────────────────────────────────────────────────
  const PAGE   = { minHeight:"100vh", background:"linear-gradient(160deg,#050d1a,#0a1525 60%,#060e1c)", fontFamily:"'Segoe UI',Arial,sans-serif", color:"#f0f4ff", margin:0 };
  const BODY   = { padding:"20px 28px", maxWidth:1100, margin:"0 auto" };
  const TABBAR = { display:"flex", gap:4, background:"rgba(255,255,255,.04)", borderRadius:10, padding:4, marginBottom:24, border:"1px solid rgba(255,255,255,.08)" };

  const tabBtn = (id: string, label: string) => (
    <button key={id} onClick={() => setTab(id)} style={{
      flex:1, padding:"10px 14px", borderRadius:8, border:"none",
      background: tab===id ? "rgba(0,255,136,.15)" : "transparent",
      color:      tab===id ? "#00ff88" : "#7a9abc",
      cursor:"pointer", fontSize:13, fontWeight: tab===id ? 800 : 500,
      fontFamily:"inherit", transition:"all .15s",
      outline: tab===id ? "1px solid rgba(0,255,136,.4)" : "none",
    }}>{label}</button>
  );

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div style={PAGE}>

      {/* HEADER */}
      <div style={{ background:"linear-gradient(90deg,rgba(0,255,136,.10),transparent 70%)", borderBottom:"2px solid rgba(0,255,136,.30)", padding:"18px 28px", display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:12 }}>
        <div>
          <div style={{ fontSize:11, color:"#00ff88", letterSpacing:5, fontWeight:700, marginBottom:2 }}>◆ PARLAY ENGINE v6</div>
          <div style={{ fontSize:26, fontWeight:900, color:"#fff" }}>AI Betting Dashboard</div>
          <div style={{ fontSize:13, color:"#7a9abc", marginTop:3 }}>Upload FanDuel screenshots → instant analysis</div>
        </div>
        <div style={{ display:"flex", gap:10, alignItems:"center" }}>
          {bets.length > 0 && (
            <div style={{ background:"rgba(0,255,136,.1)", border:"2px solid rgba(0,255,136,.3)", borderRadius:10, padding:"10px 18px", textAlign:"center" }}>
              <div style={{ fontSize:10, color:"#7a9abc", letterSpacing:3 }}>LEGS FOUND</div>
              <div style={{ fontSize:26, color:"#00ff88", fontWeight:900 }}>{bets.length}</div>
            </div>
          )}
          <div style={{ background:"rgba(255,255,255,.05)", border:"1px solid rgba(255,255,255,.1)", borderRadius:10, padding:"10px 18px", textAlign:"center" }}>
            <div style={{ fontSize:10, color:"#7a9abc", letterSpacing:3 }}>BANKROLL</div>
            <div style={{ fontSize:26, color:"#fff", fontWeight:900 }}>$1,000</div>
          </div>
        </div>
      </div>

      <div style={BODY}>

        {/* TAB BAR */}
        <div style={TABBAR}>
          {tabBtn("UPLOAD", "📤  Upload Screenshots")}
          {tabBtn("PICKS",  `📋  Tonight's Picks${bets.length ? ` (${bets.length})` : ""}`)}
          {tabBtn("BUILD",  `🎯  Parlay Builder${parlay.length ? ` (${parlay.length})` : ""}`)}
          {tabBtn("GUIDE",  "❓  How It Works")}
        </div>

        {/* ══ UPLOAD TAB ══ */}
        {tab === "UPLOAD" && (
          <div>
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                border:`2px dashed ${dragOver ? "#00ff88" : "rgba(255,255,255,.2)"}`,
                borderRadius:14, padding:"48px 20px", textAlign:"center", cursor:"pointer",
                background: dragOver ? "rgba(0,255,136,.06)" : "rgba(255,255,255,.02)",
                transition:"all .2s", marginBottom:20,
              }}
            >
              <div style={{ fontSize:48, marginBottom:12 }}>📸</div>
              <div style={{ fontSize:20, fontWeight:800, color:"#fff", marginBottom:8 }}>Drop FanDuel Screenshots Here</div>
              <div style={{ fontSize:14, color:"#7a9abc", marginBottom:16 }}>or click to browse — PNG, JPG, WEBP supported</div>
              <div style={{ display:"inline-block", padding:"10px 28px", borderRadius:8, background:"rgba(0,255,136,.15)", border:"2px solid rgba(0,255,136,.4)", color:"#00ff88", fontSize:14, fontWeight:800 }}>Choose Files</div>
            </div>
            <input ref={fileRef} type="file" accept="image/*" multiple style={{ display:"none" }} onChange={e => addFiles(e.target.files)} />

            {/* Thumbnails */}
            {images.length > 0 && (
              <div style={{ marginBottom:24 }}>
                <div style={{ fontSize:12, color:"#7a9abc", letterSpacing:3, fontWeight:700, marginBottom:12 }}>
                  {images.length} SCREENSHOT{images.length > 1 ? "S" : ""} QUEUED
                </div>
                <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
                  {images.map((img, i) => (
                    <div key={i} style={{ position:"relative" }}>
                      <img src={img.url} alt={img.name} style={{ width:110, height:80, objectFit:"cover", borderRadius:8, border:"1px solid rgba(255,255,255,.15)" }} />
                      <button onClick={() => setImages(prev => prev.filter((_,j) => j !== i))} style={{ position:"absolute", top:-6, right:-6, width:20, height:20, borderRadius:"50%", background:"#ff5555", border:"none", color:"#fff", cursor:"pointer", fontSize:11, fontWeight:900, display:"flex", alignItems:"center", justifyContent:"center" }}>✕</button>
                      <div style={{ fontSize:10, color:"#4a6a8a", marginTop:4, maxWidth:110, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{img.name}</div>
                    </div>
                  ))}
                  <div onClick={() => fileRef.current?.click()} style={{ width:110, height:80, borderRadius:8, border:"2px dashed rgba(255,255,255,.15)", display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", color:"#4a6a8a", fontSize:28, background:"rgba(255,255,255,.02)" }}>+</div>
                </div>
              </div>
            )}

            {error && (
              <div style={{ background:"rgba(255,85,85,.1)", border:"1px solid rgba(255,85,85,.4)", borderRadius:8, padding:"12px 18px", color:"#ff9999", fontSize:13, marginBottom:16 }}>⚠️ {error}</div>
            )}

            <button onClick={analyse} disabled={loading || !images.length} style={{
              width:"100%", padding:"16px", borderRadius:10, cursor: loading || !images.length ? "not-allowed" : "pointer",
              background: loading || !images.length ? "rgba(255,255,255,.07)" : "linear-gradient(90deg,rgba(0,255,136,.3),rgba(0,200,100,.2))",
              color:  loading || !images.length ? "#4a6a8a" : "#00ff88",
              fontSize:16, fontWeight:900, fontFamily:"inherit", letterSpacing:1,
              border:`2px solid ${loading || !images.length ? "rgba(255,255,255,.1)" : "rgba(0,255,136,.5)"}`,
              transition:"all .2s",
            }}>
              {loading ? "🔍  Analysing screenshots with AI…" : images.length ? `🚀  Analyse ${images.length} Screenshot${images.length > 1 ? "s" : ""}` : "📤  Upload screenshots above first"}
            </button>

            {loading && (
              <div style={{ textAlign:"center", marginTop:20, color:"#7a9abc", fontSize:13 }}>
                <div style={{ fontSize:32, marginBottom:8 }}>⚙️</div>
                Claude is reading your screenshots and extracting all player props…<br />
                <span style={{ fontSize:11, color:"#4a6a8a" }}>This takes 10–20 seconds depending on how many images you uploaded.</span>
              </div>
            )}

            {bets.length > 0 && !loading && (
              <div style={{ marginTop:20, background:"rgba(0,255,136,.06)", border:"2px solid rgba(0,255,136,.3)", borderRadius:10, padding:"14px 20px", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
                <div>
                  <div style={{ fontSize:15, fontWeight:800, color:"#00ff88" }}>✓ Analysis complete — {bets.length} legs found</div>
                  <div style={{ fontSize:12, color:"#7a9abc", marginTop:3 }}>Switch to Tonight's Picks to view results</div>
                </div>
                <button onClick={() => setTab("PICKS")} style={{ padding:"10px 20px", borderRadius:8, border:"2px solid rgba(0,255,136,.5)", background:"rgba(0,255,136,.15)", color:"#00ff88", cursor:"pointer", fontSize:13, fontWeight:800, fontFamily:"inherit" }}>View Picks →</button>
              </div>
            )}
          </div>
        )}

        {/* ══ PICKS TAB ══ */}
        {tab === "PICKS" && (
          <>
            {bets.length === 0 ? (
              <div style={{ textAlign:"center", padding:"60px 20px", color:"#4a6a8a" }}>
                <div style={{ fontSize:48, marginBottom:12 }}>📤</div>
                <div style={{ fontSize:18, fontWeight:700, color:"#7a9abc", marginBottom:8 }}>No picks yet</div>
                <div style={{ fontSize:14, marginBottom:20 }}>Upload your FanDuel screenshots first</div>
                <button onClick={() => setTab("UPLOAD")} style={{ padding:"10px 24px", borderRadius:8, border:"2px solid rgba(0,255,136,.4)", background:"rgba(0,255,136,.1)", color:"#00ff88", cursor:"pointer", fontSize:14, fontWeight:800, fontFamily:"inherit" }}>Go to Upload →</button>
              </div>
            ) : (
              <>
                {/* Tier legend */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:20 }}>
                  {Object.entries(TIER_CFG).map(([k, t]) => (
                    <div key={k} style={{ background:t.bg, border:`1px solid ${t.border}`, borderRadius:10, padding:"12px 16px" }}>
                      <div style={{ fontSize:14, color:t.color, fontWeight:800, marginBottom:3 }}>{t.label}</div>
                      <div style={{ fontSize:12, color:"#94b4d4" }}>
                        {k==="LOCK" ? "73–100% win chance" : k==="STRONG" ? "60–72% win chance" : "Underpriced by the book"}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Filter bar */}
                <div style={{ display:"flex", gap:8, marginBottom:10, flexWrap:"wrap", alignItems:"center" }}>
                  <span style={{ fontSize:12, color:"#7a9abc" }}>Show:</span>
                  {(["ALL","LOCK","STRONG","VALUE"] as const).map(f => {
                    const tc = f !== "ALL" ? TIER_CFG[f] : undefined;
                    const active = filter === f;
                    return (
                      <button key={f} onClick={() => setFilter(f)} style={{
                        padding:"7px 16px", borderRadius:6, cursor:"pointer", fontSize:12, fontWeight:700, fontFamily:"inherit",
                        border:`2px solid ${active ? (tc?.border||"rgba(0,255,136,.4)") : "rgba(255,255,255,.1)"}`,
                        background: active ? (tc?.bg||"rgba(0,255,136,.1)") : "transparent",
                        color: active ? (tc?.color||"#00ff88") : "#7a9abc",
                      }}>{f === "ALL" ? "All Picks" : tc?.label}</button>
                    );
                  })}
                </div>

                <div style={{ fontSize:12, color:"#4a6a8a", marginBottom:14 }}>
                  🔗 Click player name → opens FanDuel &nbsp;·&nbsp; ➕ Click row or button → add to parlay
                </div>

                {/* Pick cards */}
                <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
                  {filtered.map((b, i) => {
                    const tc  = TIER_CFG[b.tier] || TIER_CFG.STRONG;
                    const inP = parlay.includes(b.player);
                    return (
                      <div key={i}
                        onMouseEnter={() => setHovRow(i)} onMouseLeave={() => setHovRow(null)}
                        onClick={() => toggleLeg(b.player)}
                        style={{
                          background: inP ? "rgba(0,255,136,.07)" : hovRow===i ? "rgba(255,255,255,.04)" : "rgba(255,255,255,.025)",
                          border:`1px solid ${inP ? "rgba(0,255,136,.4)" : "rgba(255,255,255,.08)"}`,
                          borderLeft:`5px solid ${tc.color}`, borderRadius:10,
                          padding:"16px 20px", cursor:"pointer",
                          display:"grid", gridTemplateColumns:"2fr 1.1fr 90px 1fr 1fr auto",
                          alignItems:"center", gap:12, transition:"all .15s",
                        }}
                      >
                        <div>
                          <a href={fdUrl(b.player)} target="_blank" rel="noopener noreferrer"
                            onClick={e => e.stopPropagation()}
                            onMouseEnter={() => setHovLink(i)} onMouseLeave={() => setHovLink(null)}
                            style={{ fontSize:17, fontWeight:800, color: hovLink===i ? "#4da6ff" : "#ffffff", textDecoration: hovLink===i ? "underline" : "none", display:"inline-flex", alignItems:"center", gap:6, transition:"color .15s" }}
                          >
                            {b.player}
                            <span style={{ fontSize:10, background:"rgba(31,105,255,.2)", border:"1px solid rgba(31,105,255,.4)", borderRadius:4, padding:"1px 6px", color:"#4da6ff", fontWeight:700 }}>FD ↗</span>
                          </a>
                          <div style={{ fontSize:12, color:"#7a9abc", marginTop:3 }}>{b.game}</div>
                        </div>

                        <div>
                          <div style={{ fontSize:10, color:"#4a6a8a", letterSpacing:2, marginBottom:3 }}>THE BET</div>
                          <div style={{ fontSize:15, color:"#d0e4ff", fontWeight:700 }}>{b.prop}</div>
                        </div>

                        <div>
                          <div style={{ fontSize:10, color:"#4a6a8a", letterSpacing:2, marginBottom:3 }}>ODDS</div>
                          <div style={{ fontSize:22, fontWeight:900, color: b.odds > 0 ? "#fbbf24" : "#ffffff" }}>{fmtOdds(b.odds)}</div>
                          <div style={{ fontSize:10, color:"#4a6a8a", marginTop:2 }}>{oddsNote(b.odds)}</div>
                        </div>

                        <div>
                          <div style={{ fontSize:10, color:"#4a6a8a", letterSpacing:2, marginBottom:3 }}>WIN CHANCE</div>
                          <div style={{ fontSize:20, fontWeight:900, color:tc.color }}>{b.trueProb}%</div>
                          <div style={{ height:5, background:"rgba(255,255,255,.08)", borderRadius:3, marginTop:5, overflow:"hidden" }}>
                            <div style={{ height:"100%", width:`${b.trueProb}%`, background:tc.color, borderRadius:3 }} />
                          </div>
                        </div>

                        <div>
                          <div style={{ fontSize:10, color:"#4a6a8a", letterSpacing:2, marginBottom:3 }}>SUGGESTED BET</div>
                          <div style={{ fontSize:18, fontWeight:900, color:"#fff" }}>{b.kelly}</div>
                          <div style={{ fontSize:11, color:"#4a6a8a" }}>of $1,000 bankroll</div>
                        </div>

                        <button onClick={e => { e.stopPropagation(); toggleLeg(b.player); }} style={{
                          padding:"10px 14px", borderRadius:8, cursor:"pointer", fontFamily:"inherit",
                          fontSize:12, fontWeight:800, whiteSpace:"nowrap",
                          border:`2px solid ${inP ? "#00ff88" : "rgba(255,255,255,.2)"}`,
                          background: inP ? "rgba(0,255,136,.2)" : "rgba(255,255,255,.05)",
                          color: inP ? "#00ff88" : "#94b4d4", transition:"all .15s",
                        }}>{inP ? "✓ Added" : "+ Parlay"}</button>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </>
        )}

        {/* ══ BUILD TAB ══ */}
        {tab === "BUILD" && (
          <>
            <div style={{ fontSize:14, color:"#7a9abc", marginBottom:22 }}>
              Add legs from the Picks tab. All legs must win for the parlay to pay out.
            </div>

            {parlay.length === 0 ? (
              <div style={{ textAlign:"center", padding:"50px 20px", background:"rgba(255,255,255,.02)", border:"1px dashed rgba(255,255,255,.1)", borderRadius:10, color:"#4a6a8a", fontSize:14 }}>
                No legs added yet.<br />
                <button onClick={() => setTab("PICKS")} style={{ marginTop:16, padding:"10px 24px", borderRadius:8, border:"2px solid rgba(0,255,136,.4)", background:"rgba(0,255,136,.1)", color:"#00ff88", cursor:"pointer", fontSize:14, fontWeight:800, fontFamily:"inherit" }}>Go to Picks →</button>
              </div>
            ) : (
              <div style={{ background:"rgba(0,255,136,.04)", border:"2px solid rgba(0,255,136,.3)", borderRadius:12, padding:"22px 26px" }}>
                <div style={{ fontSize:12, color:"#00ff88", letterSpacing:4, fontWeight:700, marginBottom:16 }}>◆ YOUR PARLAY SLIP</div>

                {parlay.map((name, i) => {
                  const b = bets.find(x => x.player === name);
                  return b ? (
                    <div key={i} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"12px 0", borderBottom:"1px solid rgba(255,255,255,.06)" }}>
                      <div>
                        <span style={{ fontSize:15, fontWeight:800, color:"#fff" }}>{b.player}</span>
                        <span style={{ fontSize:13, color:"#7a9abc", marginLeft:10 }}>{b.prop}</span>
                        <span style={{ fontSize:12, color:"#4a6a8a", marginLeft:10 }}>{b.game}</span>
                      </div>
                      <div style={{ display:"flex", alignItems:"center", gap:14 }}>
                        <span style={{ fontSize:20, fontWeight:900, color: b.odds > 0 ? "#fbbf24" : "#fff" }}>{fmtOdds(b.odds)}</span>
                        <span style={{ fontSize:14, color: TIER_CFG[b.tier]?.color || "#fff", fontWeight:700 }}>{b.trueProb}%</span>
                        <button onClick={() => toggleLeg(name)} style={{ background:"rgba(255,85,85,.15)", border:"1px solid rgba(255,85,85,.4)", borderRadius:6, color:"#ff5555", cursor:"pointer", padding:"5px 12px", fontSize:12, fontFamily:"inherit", fontWeight:700 }}>✕ Remove</button>
                      </div>
                    </div>
                  ) : null;
                })}

                {co && (
                  <div style={{ marginTop:20, paddingTop:16, borderTop:"1px solid rgba(0,255,136,.3)", display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16, textAlign:"center" }}>
                    <div>
                      <div style={{ fontSize:11, color:"#4a6a8a", letterSpacing:2, marginBottom:4 }}>COMBINED ODDS</div>
                      <div style={{ fontSize:34, fontWeight:900, color:"#00ff88" }}>{co.american}</div>
                    </div>
                    <div>
                      <div style={{ fontSize:11, color:"#4a6a8a", letterSpacing:2, marginBottom:4 }}>$100 WINS YOU</div>
                      <div style={{ fontSize:34, fontWeight:900, color:"#fbbf24" }}>${co.win100}</div>
                    </div>
                    <div>
                      <div style={{ fontSize:11, color:"#4a6a8a", letterSpacing:2, marginBottom:4 }}>LEGS</div>
                      <div style={{ fontSize:34, fontWeight:900, color:"#c084fc" }}>{parlay.length}</div>
                    </div>
                  </div>
                )}

                <div style={{ marginTop:18, display:"flex", gap:10 }}>
                  <a href="https://sportsbook.fanduel.com" target="_blank" rel="noopener noreferrer"
                    style={{ flex:1, textAlign:"center", padding:"13px 0", borderRadius:8, background:"rgba(0,255,136,.2)", border:"2px solid rgba(0,255,136,.5)", color:"#00ff88", fontSize:14, fontWeight:800, textDecoration:"none", display:"block" }}>
                    Open FanDuel to Place Bet ↗
                  </a>
                  <button onClick={() => setParlay([])} style={{ padding:"13px 20px", borderRadius:8, border:"1px solid rgba(255,85,85,.4)", background:"rgba(255,85,85,.1)", color:"#ff5555", cursor:"pointer", fontSize:13, fontWeight:700, fontFamily:"inherit" }}>
                    Clear All
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {/* ══ GUIDE TAB ══ */}
        {tab === "GUIDE" && (
          <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
            {[
              { q:"How does the upload work?",            a:"Upload FanDuel screenshots. Claude's vision AI reads every visible player prop, extracts the player name, bet type, and odds, then runs the full algorithm to rank them automatically." },
              { q:"What is a Prop Bet?",                  a:"Instead of betting who wins the game, you bet on one player's stats — like scoring 20+ points or recording 6+ assists." },
              { q:"What do the odds numbers mean?",       a:"Negative (e.g. -180) = likely to happen, smaller payout. Positive (e.g. +122) = less likely, bigger payout. -180 means bet $180 to win $100. +122 means bet $100 to win $122." },
              { q:"What is Win Chance %?",                a:"The true probability after removing the bookmaker's built-in margin (called the 'vig'). A 75% chance means this bet hits roughly 3 out of every 4 times." },
              { q:"What is Suggested Bet?",               a:"Dollar amount recommended from your $1,000 bankroll using the Kelly Criterion — a math formula that sizes your bet based on your edge. Bigger edge = bigger suggested bet." },
              { q:"What is a Parlay?",                    a:"Multiple bets chained together. Every single leg must win. Riskier, but pays much more. A 4-leg parlay might return 20x your stake." },
              { q:"What does 🔒 Lock vs 💎 Value mean?",  a:"LOCK = high probability (73%+), usually negative odds — safer. VALUE = lower probability but the book underpriced the odds — highest mathematical edge even if it feels riskier." },
              { q:"What is Expected Value (EV)?",         a:"EV tells you if a bet is profitable long-term. Positive EV means the book underestimated the real probability — you have a mathematical edge on that bet." },
            ].map((item, i) => (
              <div key={i} style={{ background:"rgba(255,255,255,.03)", border:"1px solid rgba(255,255,255,.08)", borderRadius:10, padding:"16px 20px" }}>
                <div style={{ fontSize:15, fontWeight:800, color:"#fff", marginBottom:8 }}>❓ {item.q}</div>
                <div style={{ fontSize:14, color:"#94b4d4", lineHeight:1.7 }}>{item.a}</div>
              </div>
            ))}
          </div>
        )}

        {/* FOOTER WARNING */}
        <div style={{ marginTop:28, background:"rgba(255,85,85,.07)", border:"1px solid rgba(255,85,85,.25)", borderRadius:8, padding:"14px 20px", display:"flex", alignItems:"center", gap:12 }}>
          <span style={{ fontSize:20 }}>⚠️</span>
          <div>
            <div style={{ fontSize:13, color:"#ff9999", fontWeight:800, marginBottom:3 }}>Always Verify Before Betting</div>
            <div style={{ fontSize:12, color:"#7a9abc" }}>Check injury reports · Lines move fast · This tool is for analysis only, not financial advice</div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default ParlayEngine;
