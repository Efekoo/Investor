(() => {
  "use strict";

  const LC = window.LightweightCharts;
  const $ = (sel) => document.querySelector(sel);
  const TZ_SHIFT = -new Date().getTimezoneOffset() * 60; // grafiklerde yerel saat göster
  const COLORS = { up: "#22c55e", down: "#ef4444", liq: "#ff3b5c", entry: "#94a3b8", grid: "#16202c", text: "#8b98a8", line: "#1f2b3a", accent: "#3b82f6", warn: "#f59e0b" };

  const state = {
    snap: null,
    levels: [],
    symbol: null,
    tf: "1m",
    alertPct: 5,
    candleReq: 0,
    lastCandleTime: 0,
    priceLines: [],
    priceLineKey: "",
    tradesCount: -1,
    ws: null,
    pollTimer: null,
  };

  // ── Biçimlendirme ────────────────────────────────────────────────────
  const nf = (d) => new Intl.NumberFormat("tr-TR", { minimumFractionDigits: d, maximumFractionDigits: d });
  const money = (v) => (v < 0 ? "-$" : "$") + nf(2).format(Math.abs(v));
  const signedMoney = (v) => (v > 0 ? "+" : v < 0 ? "-" : "") + "$" + nf(2).format(Math.abs(v));
  // Türkçe yazım: % işareti sayının önünde (+%1,25 / -%0,80 / %30)
  const pct = (v, d = 2) => (v > 0 ? "+" : v < 0 ? "-" : "") + "%" + nf(d).format(Math.abs(v));
  const pctU = (v, d = 2) => (v < 0 ? "-" : "") + "%" + nf(d).format(Math.abs(v));
  const decimalsFor = (p) => (p >= 1000 ? 2 : p >= 10 ? 3 : p >= 1 ? 4 : p >= 0.1 ? 5 : 6);
  const price = (p, ref) => (p ? nf(decimalsFor(ref || p)).format(p) : "—");
  const qty = (q) => nf(q >= 100 ? 2 : q >= 1 ? 4 : 6).format(q);
  const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ago = (sec) => (sec < 60 ? `${Math.round(sec)} sn` : sec < 3600 ? `${Math.round(sec / 60)} dk` : sec < 86400 ? `${Math.round(sec / 3600)} sa` : `${Math.round(sec / 86400)} gün`);
  const timeStr = (iso) => {
    const d = new Date(iso);
    return isNaN(d) ? "—" : d.toLocaleString("tr-TR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  };

  // ── API ──────────────────────────────────────────────────────────────
  function token() {
    try { return localStorage.getItem("dashboardToken") || ""; } catch { return ""; }
  }
  function saveToken(t) {
    try { localStorage.setItem("dashboardToken", t); } catch { /* depolama kapalı */ }
  }

  async function api(path, opts = {}, retried = false) {
    const headers = { "Content-Type": "application/json" };
    const t = token();
    if (t) headers["X-Dashboard-Token"] = t;
    const res = await fetch(path, { ...opts, headers });
    if (res.status === 401 && !retried) {
      const entered = window.prompt("Panel token'ı (DASHBOARD_TOKEN):");
      if (entered) { saveToken(entered.trim()); return api(path, opts, true); }
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  }

  let noticeTimer = null;
  function notice(msg, isError = false, ms = 6000) {
    const el = $("#notice");
    el.textContent = msg;
    el.classList.toggle("error", isError);
    el.hidden = !msg;
    clearTimeout(noticeTimer);
    if (msg && ms) noticeTimer = setTimeout(() => (el.hidden = true), ms);
  }

  // ── Grafikler ────────────────────────────────────────────────────────
  const baseChartOpts = {
    autoSize: true,
    layout: { background: { type: "solid", color: "transparent" }, textColor: COLORS.text, fontFamily: "Inter, system-ui, sans-serif", fontSize: 11 },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderColor: COLORS.line },
    timeScale: { borderColor: COLORS.line, timeVisible: true, secondsVisible: false },
    crosshair: { mode: LC.CrosshairMode.Normal },
  };

  const priceChart = LC.createChart($("#price-chart"), baseChartOpts);
  const candleSeries = priceChart.addCandlestickSeries({
    upColor: COLORS.up, downColor: COLORS.down, borderVisible: false, wickUpColor: COLORS.up, wickDownColor: COLORS.down,
    // Pozisyon seviyeleri (giriş/SL/TP, yakınsa likidasyon) görünür alanda kalsın
    autoscaleInfoProvider: (original) => {
      const res = original();
      if (!res || !state.levels.length) return res;
      res.priceRange.minValue = Math.min(res.priceRange.minValue, ...state.levels);
      res.priceRange.maxValue = Math.max(res.priceRange.maxValue, ...state.levels);
      return res;
    },
  });
  const volumeSeries = priceChart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "", lastValueVisible: false, priceLineVisible: false });
  volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

  const equityChart = LC.createChart($("#equity-chart"), { ...baseChartOpts, handleScroll: false, handleScale: false });
  const equitySeries = equityChart.addAreaSeries({
    lineColor: COLORS.accent, topColor: "rgba(59,130,246,.28)", bottomColor: "rgba(59,130,246,0)", lineWidth: 2,
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });

  const toBar = (c) => ({ time: c.time + TZ_SHIFT, open: c.open, high: c.high, low: c.low, close: c.close });
  const toVol = (c) => ({ time: c.time + TZ_SHIFT, value: c.volume, color: c.close >= c.open ? "rgba(34,197,94,.35)" : "rgba(239,68,68,.35)" });

  function setLegend(bar) {
    const el = $("#chart-legend");
    if (!bar) { el.textContent = ""; return; }
    const ref = bar.close;
    const chg = ((bar.close - bar.open) / bar.open) * 100;
    el.innerHTML = `A ${price(bar.open, ref)} Y ${price(bar.high, ref)} D ${price(bar.low, ref)} K ${price(bar.close, ref)} <span class="${cls(chg)}">${pct(chg)}</span>`;
  }
  priceChart.subscribeCrosshairMove((param) => {
    const bar = param && param.seriesData ? param.seriesData.get(candleSeries) : null;
    setLegend(bar || state.lastBar);
  });

  async function loadCandles(full) {
    if (!state.symbol) return;
    const sym = state.symbol;
    const tf = state.tf;
    const req = full ? ++state.candleReq : state.candleReq;
    const limit = full ? 500 : 3;
    try {
      const data = await api(`/api/candles?symbol=${encodeURIComponent(sym)}&timeframe=${tf}&limit=${limit}`);
      // Bu arada sembol/zaman dilimi değiştiyse ya da yeni bir tam yükleme başladıysa sonucu at
      if (sym !== state.symbol || tf !== state.tf || req !== state.candleReq) return;
      if (!full && !state.lastCandleTime) return;
      const bars = data.candles.map(toBar);
      if (!bars.length) return;
      if (full) {
        const p = bars[bars.length - 1].close;
        const d = decimalsFor(p);
        candleSeries.applyOptions({ priceFormat: { type: "price", precision: d, minMove: 1 / 10 ** d } });
        candleSeries.setData(bars);
        volumeSeries.setData(data.candles.map(toVol));
        priceChart.timeScale().scrollToRealTime();
      } else {
        data.candles.forEach((c, i) => {
          if (bars[i].time >= state.lastCandleTime) {
            candleSeries.update(bars[i]);
            volumeSeries.update(toVol(c));
          }
        });
      }
      state.lastCandleTime = bars[bars.length - 1].time;
      state.lastBar = bars[bars.length - 1];
      setLegend(state.lastBar);
    } catch (err) {
      if (full) notice(`Grafik verisi alınamadı: ${err.message}`, true);
    }
  }

  function updatePriceLines() {
    const pos = state.snap ? state.snap.positions.find((p) => p.symbol === state.symbol) : null;
    const key = pos ? [pos.symbol, pos.entry_price, pos.stop_loss, pos.take_profit, pos.liquidation_price].join("|") : "";
    if (key === state.priceLineKey) return;
    state.priceLineKey = key;
    state.priceLines.forEach((l) => candleSeries.removePriceLine(l));
    state.priceLines = [];
    state.levels = [];
    $("#liq-note").textContent = "";
    if (!pos) return;
    state.levels = [pos.entry_price, pos.stop_loss, pos.take_profit].filter((v) => v > 0);
    // Likidasyon çok uzaktaysa grafiği sıkıştırmasın; yalnızca notta göster
    const liqNear = pos.liquidation_price > 0 && Math.abs(pos.liquidation_price - pos.mark_price) / pos.mark_price <= 0.12;
    if (liqNear) state.levels.push(pos.liquidation_price);
    else if (pos.liquidation_price > 0) $("#liq-note").textContent = `Likidasyon ${price(pos.liquidation_price, pos.mark_price)} (grafik dışında)`;
    const add = (value, color, title, style) => {
      if (value > 0) state.priceLines.push(candleSeries.createPriceLine({ price: value, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title }));
    };
    add(pos.entry_price, COLORS.entry, `Giriş ${pos.side}`, LC.LineStyle.Dotted);
    add(pos.stop_loss, COLORS.down, "SL", LC.LineStyle.Solid);
    add(pos.take_profit, COLORS.up, "TP", LC.LineStyle.Solid);
    if (liqNear) add(pos.liquidation_price, COLORS.liq, `Likidasyon ${pos.leverage}x`, LC.LineStyle.Dashed);
  }

  function selectSymbol(sym) {
    if (!sym || sym === state.symbol) return;
    state.symbol = sym;
    state.lastCandleTime = 0;
    state.priceLineKey = "__";
    document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.symbol === sym)));
    loadCandles(true);
    updatePriceLines();
  }

  // ── Render ───────────────────────────────────────────────────────────
  function renderHeader(s) {
    const mode = $("#mode-badge");
    mode.textContent = s.mode.toUpperCase();
    mode.className = `badge ${s.mode === "live" ? "live" : "paper"}`;
    const lev = $("#lev-badge");
    lev.hidden = !s.leverage.enabled;
    lev.textContent = `${s.leverage.leverage}x ${s.leverage.margin_mode === "isolated" ? "izole" : "çapraz"}`;

    const st = $("#bot-status");
    const text = st.querySelector(".status-text");
    if (!s.state_found) {
      st.className = "status down";
      text.textContent = "Bot durum dosyası yok";
    } else if (s.state_age_seconds !== null && s.state_age_seconds <= 150) {
      st.className = "status ok";
      text.textContent = `Bot çalışıyor · ${ago(s.state_age_seconds)} önce`;
    } else {
      st.className = "status stale";
      text.textContent = `Bot durmuş olabilir · son güncelleme ${ago(s.state_age_seconds || 0)} önce`;
    }
    const kill = $("#kill-toggle");
    if (!kill.dataset.busy) kill.checked = s.kill_switch;
  }

  function renderKpis(s) {
    const a = s.summary;
    const t = s.trade_stats;
    $("#k-equity").textContent = money(a.equity);
    $("#k-equity-sub").textContent = `nakit + teminat + açık PnL`;
    $("#k-cash").textContent = money(a.cash);
    $("#k-margin").textContent = s.leverage.enabled ? money(a.margin_used) : "Kaldıraç kapalı";
    $("#k-margin-sub").textContent = s.leverage.enabled ? `varlığın ${pctU(a.margin_usage_pct, 1)} kadarı` : "";
    const bar = $("#k-margin-bar");
    bar.style.width = `${Math.min(100, a.margin_usage_pct)}%`;
    bar.style.background = a.margin_usage_pct >= 80 ? COLORS.down : a.margin_usage_pct >= 50 ? COLORS.warn : COLORS.accent;
    const upnl = $("#k-upnl");
    upnl.textContent = signedMoney(a.unrealized_pnl);
    upnl.className = `kpi-value ${cls(a.unrealized_pnl)}`;
    $("#k-open").textContent = `${a.open_positions} açık pozisyon`;
    const rpnl = $("#k-rpnl");
    rpnl.textContent = signedMoney(t.total_pnl);
    rpnl.className = `kpi-value ${cls(t.total_pnl)}`;
    $("#k-stats").textContent = t.count
      ? `${t.count} işlem · ${pctU(t.win_rate, 0)} kazanç${t.liquidations ? ` · ${t.liquidations} likidasyon` : ""}`
      : "henüz işlem yok";
  }

  function renderAlerts(s) {
    const el = $("#alerts");
    if (!s.alerts.length) { el.hidden = true; return; }
    el.hidden = false;
    el.innerHTML = "<strong>Likidasyon riski:</strong> " + s.alerts
      .map((p) => `${esc(p.symbol)} ${p.side} ${p.leverage}x — likidasyona ${pctU(p.liq_distance_pct)} (${price(p.liquidation_price, p.mark_price)})`)
      .join(" · ");
  }

  function renderTabs(s) {
    const wrap = $("#symbol-tabs");
    const key = s.symbols.join(",");
    if (wrap.dataset.key !== key) {
      wrap.dataset.key = key;
      wrap.innerHTML = s.symbols
        .map((sym) => `<button class="tab" role="tab" data-symbol="${esc(sym)}" aria-selected="false"><span class="pos-dot"></span>${esc(sym)}</button>`)
        .join("");
      wrap.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => selectSymbol(b.dataset.symbol)));
      if (!state.symbol || !s.symbols.includes(state.symbol)) {
        state.symbol = null;
        selectSymbol(s.symbols[0]);
      } else {
        wrap.querySelector(`[data-symbol="${CSS.escape(state.symbol)}"]`).setAttribute("aria-selected", "true");
      }
    }
    const open = Object.fromEntries(s.positions.map((p) => [p.symbol, p.side]));
    wrap.querySelectorAll(".tab").forEach((b) => {
      const side = open[b.dataset.symbol];
      b.querySelector(".pos-dot").style.background = side === "LONG" ? COLORS.up : side === "SHORT" ? COLORS.down : "transparent";
      b.title = side ? `Açık ${side} pozisyon` : "";
    });
  }

  function distCell(p) {
    if (p.liq_distance_pct === null) return '<span class="muted">Yok (1x)</span>';
    const d = p.liq_distance_pct;
    const color = d <= state.alertPct ? COLORS.liq : d <= state.alertPct * 2 ? COLORS.warn : COLORS.up;
    const width = Math.max(3, Math.min(100, (d / 35) * 100));
    return `<div class="dist"><div class="bar"><span style="width:${width}%;background:${color}"></span></div><span class="val" style="color:${color}">${pctU(d)}</span></div>`;
  }

  function renderPositions(s) {
    const body = $("#positions tbody");
    $("#positions-empty").hidden = s.positions.length > 0;
    $("#close-all").disabled = s.positions.length === 0;
    body.innerHTML = s.positions.map((p) => {
      const risky = p.liq_distance_pct !== null && p.liq_distance_pct <= state.alertPct;
      const ref = p.mark_price;
      return `<tr class="${risky ? "risky" : ""}">
        <td><button class="sym-link" data-symbol="${esc(p.symbol)}">${esc(p.symbol)}</button><span class="sub">${esc(timeStr(p.opened_at))}</span></td>
        <td><span class="pill ${p.side === "LONG" ? "long" : "short"}">${p.side}</span>${p.leverage > 1 ? `<span class="pill lev">${p.leverage}x</span>` : ""}</td>
        <td class="num">${qty(p.qty)}<span class="sub">${money(p.notional)}</span></td>
        <td class="num">${price(p.entry_price, ref)}</td>
        <td class="num">${price(p.mark_price, ref)}</td>
        <td class="num ${cls(p.unrealized_pnl)}">${signedMoney(p.unrealized_pnl)}</td>
        <td class="num ${cls(p.roe_pct)}">${pct(p.roe_pct)}</td>
        <td class="num">${p.margin > 0 ? money(p.margin) : '<span class="muted">—</span>'}</td>
        <td class="num">${p.liquidation_price > 0 ? price(p.liquidation_price, ref) : '<span class="muted">—</span>'}</td>
        <td>${distCell(p)}</td>
        <td class="num"><span class="down">${price(p.stop_loss, ref)}</span><span class="sub up">${price(p.take_profit, ref)}</span></td>
        <td><button class="btn btn-danger-ghost btn-close" data-close="${esc(p.symbol)}">Kapat</button></td>
      </tr>`;
    }).join("");
    body.querySelectorAll(".sym-link").forEach((b) => b.addEventListener("click", () => {
      selectSymbol(b.dataset.symbol);
      $("#price-chart").scrollIntoView({ behavior: "smooth", block: "center" });
    }));
    body.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => closePosition(b.dataset.close, b)));
  }

  function renderPending(s) {
    if (!s.pending_commands) return;
    const live = s.state_age_seconds !== null && s.state_age_seconds <= 150;
    notice(live
      ? "Komut bota iletildi, işleniyor…"
      : "Komut kuyrukta, ama bot çalışmıyor görünüyor. Bot başladığında işlenecek.", !live, 0);
  }

  function render(s) {
    const firstPending = state.snap && state.snap.pending_commands > 0 && s.pending_commands === 0;
    state.snap = s;
    renderHeader(s);
    renderKpis(s);
    renderAlerts(s);
    renderTabs(s);
    renderPositions(s);
    updatePriceLines();
    renderPending(s);
    if (firstPending) notice("Komut bot tarafından işlendi.", false, 4000);
    if (s.trade_stats.count !== state.tradesCount) {
      state.tradesCount = s.trade_stats.count;
      loadTrades();
    }
  }

  async function loadTrades() {
    try {
      const data = await api("/api/trades?limit=50");
      const body = $("#trades tbody");
      $("#trades-empty").hidden = data.trades.length > 0;
      $("#trades-note").textContent = data.trades.length ? `son ${data.trades.length}` : "";
      body.innerHTML = data.trades.map((t) => {
        const side = String(t.side).toUpperCase() === "BUY" ? "LONG" : "SHORT";
        const ref = t.entry_price;
        return `<tr>
          <td>${esc(timeStr(t.closed_at))}</td>
          <td>${esc(t.symbol)}</td>
          <td><span class="pill ${side === "LONG" ? "long" : "short"}">${side}</span>${t.leverage > 1 ? `<span class="pill lev">${t.leverage}x</span>` : ""}${t.liquidated ? '<span class="pill liq">LİKİDE</span>' : ""}</td>
          <td class="num">${price(t.entry_price, ref)} → ${price(t.exit_price, ref)}</td>
          <td class="num ${cls(t.pnl)}">${signedMoney(t.pnl)}</td>
          <td class="muted strategy" title="${esc(t.strategy_name || "")}">${esc(t.strategy_name || "—")}</td>
        </tr>`;
      }).join("");
    } catch (err) { /* sessiz: bir sonraki turda tekrar denenir */ }
  }

  async function loadEquity() {
    try {
      const data = await api("/api/equity");
      $("#equity-note").textContent = data.error || (data.points.length ? `${data.points.length} nokta` : "henüz veri yok");
      equitySeries.setData(data.points.map((p) => ({ time: p.time + TZ_SHIFT, value: p.value })));
      equityChart.timeScale().fitContent();
    } catch (err) {
      $("#equity-note").textContent = "yüklenemedi";
    }
  }

  // ── Kontroller ───────────────────────────────────────────────────────
  async function closePosition(symbol, btn) {
    const p = state.snap.positions.find((x) => x.symbol === symbol);
    const detail = p ? `\n${p.side} ${qty(p.qty)} · PnL ${signedMoney(p.unrealized_pnl)}` : "";
    if (!window.confirm(`${symbol} pozisyonu piyasa fiyatından kapatılsın mı?${detail}`)) return;
    btn.disabled = true;
    btn.classList.add("pending");
    btn.textContent = "Gönderildi";
    try {
      await api(`/api/positions/${encodeURIComponent(symbol)}/close`, { method: "POST" });
      notice(`${symbol} kapatma komutu gönderildi.`);
    } catch (err) {
      notice(`Kapatılamadı: ${err.message}`, true);
      btn.disabled = false;
      btn.classList.remove("pending");
      btn.textContent = "Kapat";
    }
  }

  $("#close-all").addEventListener("click", async () => {
    const n = state.snap ? state.snap.positions.length : 0;
    if (!n || !window.confirm(`${n} açık pozisyonun tamamı piyasa fiyatından kapatılsın mı?`)) return;
    try {
      await api("/api/positions/close-all", { method: "POST" });
      notice("Tüm pozisyonlar için kapatma komutu gönderildi.");
    } catch (err) { notice(`Gönderilemedi: ${err.message}`, true); }
  });

  $("#kill-toggle").addEventListener("change", async (e) => {
    const el = e.target;
    const enabled = el.checked;
    if (!enabled && !window.confirm("Kill switch kapatılsın mı? Bot yeniden pozisyon açabilecek.")) { el.checked = true; return; }
    el.dataset.busy = "1";
    try {
      const res = await api("/api/kill-switch", { method: "POST", body: JSON.stringify({ enabled }) });
      el.checked = res.kill_switch;
      notice(res.kill_switch ? "Kill switch açık: bot yeni pozisyon açmayacak. Açık pozisyonlar SL/TP ile yönetilmeye devam ediyor." : "Kill switch kapatıldı.");
    } catch (err) {
      el.checked = !enabled;
      notice(`Kill switch değiştirilemedi: ${err.message}`, true);
    } finally {
      delete el.dataset.busy;
    }
  });

  $("#tf-select").addEventListener("change", (e) => {
    state.tf = e.target.value;
    state.lastCandleTime = 0;
    loadCandles(true);
  });

  $("#alert-pct").addEventListener("change", (e) => {
    state.alertPct = Number(e.target.value);
    try { localStorage.setItem("alertPct", String(state.alertPct)); } catch { /* yok say */ }
    connect(true);
  });

  // ── Canlı bağlantı: WebSocket, düşerse REST polling ─────────────────
  async function poll() {
    try { render(await api(`/api/snapshot?alert_pct=${state.alertPct}`)); } catch { /* sonraki tur */ }
  }

  function connect(restart = false) {
    if (restart && state.ws) { state.ws.onclose = null; state.ws.close(); }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws?alert_pct=${state.alertPct}`);
    state.ws = ws;
    ws.onopen = () => { clearInterval(state.pollTimer); state.pollTimer = null; };
    ws.onmessage = (ev) => render(JSON.parse(ev.data));
    ws.onclose = () => {
      if (!state.pollTimer) state.pollTimer = setInterval(poll, 3000);
      setTimeout(() => connect(), 5000);
    };
  }

  // ── Başlat ───────────────────────────────────────────────────────────
  try {
    const saved = Number(localStorage.getItem("alertPct"));
    if ([2, 3, 5, 10].includes(saved)) { state.alertPct = saved; $("#alert-pct").value = String(saved); }
  } catch { /* yok say */ }

  poll();
  connect();
  loadEquity();
  setInterval(() => loadCandles(false), 5000);
  setInterval(loadEquity, 30000);
})();
