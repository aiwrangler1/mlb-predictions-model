// MLB Predictions Model v2 - Frontend Application
// Now with market odds, park factors, platoon splits, form, workload, weather

const API = '';
let gamesData = [];
let projectionsData = [];
let currentSort = { col: null, dir: 'desc' };

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('dateDisplay').textContent = new Date().toLocaleDateString('en-US', {
        weekday: 'short', month: 'short', day: 'numeric', year: 'numeric'
    });
    loadSchedule();
    loadEdges();
});

function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
}

function refreshData() {
    fetch(`${API}/api/cache/clear`, { method: 'POST' }).then(() => {
        loadSchedule();
        loadEdges();
    });
}

async function setOddsApiKey() {
    const key = document.getElementById('oddsApiKey').value.trim();
    if (!key) return;
    await fetch(`${API}/api/settings/odds-api-key`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key })
    });
    document.getElementById('oddsApiKey').value = '';
    document.getElementById('oddsApiKey').placeholder = 'Key set ✓';
    loadEdges();
}

// ─── Helpers ──────────────────────────────────────────────────
function parkBadge(park) {
    if (!park) return '';
    const hr = park.hr || 1.0;
    if (hr > 1.10) return `<span class="adj-badge park-up">HR+ ${park.name || ''}</span>`;
    if (hr < 0.90) return `<span class="adj-badge park-down">HR− ${park.name || ''}</span>`;
    return `<span class="adj-badge">${park.name || ''}</span>`;
}

function weatherBadge(w) {
    if (!w || !w.temp) return '';
    let cls = '';
    let txt = `${w.temp}°F`;
    if (w.is_dome) return '<span class="adj-badge dome">Dome</span>';
    if (w.temp < 55) { cls = 'cold'; txt += ' Cold'; }
    else if (w.temp > 85) { cls = 'hot'; txt += ' Hot'; }
    if (w.hr_factor > 1.05) txt += ' HR↑';
    else if (w.hr_factor < 0.95) txt += ' HR↓';
    return `<span class="adj-badge weather ${cls}">${txt}</span>`;
}

function workloadBadge(wl) {
    if (!wl || wl.workload_status === 'unknown') return '';
    const s = wl.workload_status;
    let label = `${wl.days_rest || '?'}d rest`;
    if (wl.avg_pitches_last3) label += ` · ${wl.avg_pitches_last3}P avg`;
    return `<span class="workload-badge ${s}">${s} (${label})</span>`;
}

function bpStatusBadge(bp) {
    if (!bp) return '?';
    const s = bp.status || 'unknown';
    const colors = { full_strength: 'var(--green)', slight_thin: 'var(--text-muted)', thin: 'var(--amber)', depleted: 'var(--red)', emergency: 'var(--red)' };
    return `<span style="color:${colors[s] || 'var(--text-muted)'}">${s.replace('_',' ')}</span>`;
}

function bpDetails(bp, team) {
    if (!bp || !bp.unavailable?.length) return '';
    return bp.unavailable.map(a => `${a.name}(${a.reason})`).join(', ');
}

function formBadge(trend) {
    if (trend === 'hot') return '<span class="adj-badge hot">🔥 Hot</span>';
    if (trend === 'cold') return '<span class="adj-badge cold">❄ Cold</span>';
    return '';
}

// ─── Schedule / Slate ─────────────────────────────────────────
async function loadSchedule() {
    const grid = document.getElementById('gamesGrid');
    grid.innerHTML = '<div class="loading">Loading today\'s games...</div>';
    try {
        const res = await fetch(`${API}/api/schedule`);
        const data = await res.json();
        gamesData = data.games;
        document.getElementById('slateCount').textContent = `${data.games.length} games`;
        if (data.games.length === 0) {
            grid.innerHTML = '<div class="empty-state"><p>No games scheduled today</p></div>';
            return;
        }
        grid.innerHTML = data.games.map(g => renderGameCard(g)).join('');
    } catch (err) {
        grid.innerHTML = `<div class="empty-state"><p>Error: ${err.message}</p></div>`;
    }
}

function renderGameCard(g) {
    const awayPct = (g.away_win_prob * 100).toFixed(1);
    const homePct = (g.home_win_prob * 100).toFixed(1);
    const awayW = g.away_win_prob * 100;
    const homeW = g.home_win_prob * 100;
    const favSide = g.home_win_prob > g.away_win_prob ? 'home' : 'away';

    return `
    <div class="game-card">
        <div class="game-matchup">
            <div class="team-col">
                <div class="team-abbrev">${g.away_team}</div>
                <div class="team-name">${g.away_team_name}</div>
                <div class="prob-val away">${awayPct}%</div>
            </div>
            <div class="vs-col">
                <div class="vs-label">vs</div>
                <div class="win-prob-bar">
                    <div class="win-prob-away" style="width:${awayW}%"></div>
                    <div class="win-prob-home" style="width:${homeW}%"></div>
                </div>
                <div class="prob-display">
                    <span class="odds-chip ${favSide === 'away' ? 'favorite' : 'underdog'}">${g.away_fair_odds}</span>
                    <span class="odds-chip ${favSide === 'home' ? 'favorite' : 'underdog'}">${g.home_fair_odds}</span>
                </div>
            </div>
            <div class="team-col">
                <div class="team-abbrev">${g.home_team}</div>
                <div class="team-name">${g.home_team_name}</div>
                <div class="prob-val home">${homePct}%</div>
            </div>
        </div>
        <div class="game-pitchers">
            <div class="pitcher-info">
                <span class="pitcher-name">${g.away_pitcher_name}</span>
            </div>
            <div class="pitcher-info" style="text-align:right">
                <span class="pitcher-name">${g.home_pitcher_name}</span>
            </div>
        </div>
        <div class="adj-badges">
            ${parkBadge(g.park)}
            ${weatherBadge(g.weather)}
        </div>
        <div class="game-footer">
            <span class="meta">${g.status || 'Scheduled'}</span>
            <button class="btn btn-sm btn-primary" onclick="simulateGame(${g.game_id})">Simulate</button>
        </div>
    </div>`;
}

// ─── Simulation ───────────────────────────────────────────────
async function simulateGame(gameId) {
    switchTab('simulate');
    const container = document.getElementById('simResults');
    container.innerHTML = '<div class="loading">Running 5,000 Monte Carlo sims with park factors, platoon splits, form, workload & weather...</div>';
    try {
        const res = await fetch(`${API}/api/simulate/${gameId}?n_sims=5000`);
        const data = await res.json();
        renderSimResults(data);
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Simulation error: ${err.message}</p></div>`;
    }
}

function renderSimResults(data) {
    const g = data.game;
    const adj = data.adjustments || {};
    const edges = data.edges || {};
    const awayPct = (data.away_win_pct * 100).toFixed(1);
    const homePct = (data.home_win_pct * 100).toFixed(1);
    const awayHigh = data.away_win_pct >= data.home_win_pct;

    const container = document.getElementById('simResults');
    container.innerHTML = `
        <!-- Top Stats -->
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">${g.away_team} Win %</div>
                <div class="stat-value" style="color:${awayHigh ? 'var(--green)' : 'var(--text-muted)'}">${awayPct}%</div>
                <div class="stat-sub">${data.away_runs_mean.toFixed(2)} avg runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">${g.home_team} Win %</div>
                <div class="stat-value" style="color:${!awayHigh ? 'var(--green)' : 'var(--text-muted)'}">${homePct}%</div>
                <div class="stat-sub">${data.home_runs_mean.toFixed(2)} avg runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Runs</div>
                <div class="stat-value" style="color:var(--amber)">${data.total_runs_mean.toFixed(1)}</div>
                <div class="stat-sub">med: ${data.total_runs_median.toFixed(1)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Simulations</div>
                <div class="stat-value" style="color:var(--accent)">${data.n_sims.toLocaleString()}</div>
                <div class="stat-sub">Monte Carlo</div>
            </div>
        </div>

        <!-- Adjustments Summary -->
        <div class="sim-panel">
            <h3>Model Adjustments Applied</h3>
            <div class="stat-row" style="margin-bottom:0">
                <div class="stat-card">
                    <div class="stat-label">Data / Lineup</div>
                    <div class="stat-sub">${adj.data_blend || 'Steamer + 2025'}</div>
                    <div class="stat-sub" style="color:${adj.lineup_source === 'actual' ? 'var(--green)' : 'var(--amber)'}">
                        Lineup: ${adj.lineup_source === 'actual' ? 'Actual (confirmed)' : 'Projected'}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Park</div>
                    <div class="stat-sub">${adj.park?.name || '?'}</div>
                    <div class="stat-sub">HR: ${adj.park?.hr?.toFixed(2) || '1.00'}x | Runs: ${adj.park?.runs?.toFixed(2) || '1.00'}x</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Weather</div>
                    <div class="stat-sub">${adj.weather?.description || 'No data'}</div>
                    <div class="stat-sub">HR: ${adj.weather?.hr_factor?.toFixed(2) || '1.00'}x | Hit: ${adj.weather?.hit_factor?.toFixed(2) || '1.00'}x</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">HP Umpire</div>
                    <div class="stat-sub">${adj.umpire?.name || 'Unknown'}</div>
                    <div class="stat-sub" style="color:${adj.umpire?.run_impact > 0.1 ? 'var(--green)' : adj.umpire?.run_impact < -0.1 ? 'var(--red)' : 'var(--text-muted)'}">
                        ${adj.umpire?.zone || 'neutral'} zone | ${adj.umpire?.run_impact > 0 ? '+' : ''}${adj.umpire?.run_impact?.toFixed(2) || '0'} runs
                    </div>
                </div>
            </div>
            <div class="stat-row" style="margin-bottom:0;margin-top:8px">
                <div class="stat-card">
                    <div class="stat-label">Bullpen</div>
                    <div class="stat-sub">${g.away_team} BP: ${adj.bullpen?.away?.era || '?'} ERA | ${g.home_team} BP: ${adj.bullpen?.home?.era || '?'} ERA</div>
                    <div class="stat-sub">SP exit: ~${adj.bullpen?.away_sp_ip || '?'} / ~${adj.bullpen?.home_sp_ip || '?'} IP</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">BP Availability</div>
                    <div class="stat-sub">${g.away_team}: ${bpStatusBadge(adj.bullpen_availability?.away)} | ${g.home_team}: ${bpStatusBadge(adj.bullpen_availability?.home)}</div>
                    <div class="stat-sub">${bpDetails(adj.bullpen_availability?.away, g.away_team)} ${bpDetails(adj.bullpen_availability?.home, g.home_team)}</div>
                </div>
            </div>
            <div class="stat-row" style="margin-bottom:0;margin-top:8px">
                <div class="stat-card">
                    <div class="stat-label">${g.away_team} SP</div>
                    <div class="stat-sub">${data.away_pitcher?.name || 'TBD'} (${data.away_pitcher?.hand || '?'}HP)</div>
                    <div class="stat-sub">${workloadBadge(data.away_pitcher?.workload)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">${g.home_team} SP</div>
                    <div class="stat-sub">${data.home_pitcher?.name || 'TBD'} (${data.home_pitcher?.hand || '?'}HP)</div>
                    <div class="stat-sub">${workloadBadge(data.home_pitcher?.workload)}</div>
                </div>
            </div>
        </div>

        ${edges.away ? `
        <div class="sim-panel">
            <h3>Market Edge</h3>
            <div class="stat-row" style="margin-bottom:0">
                <div class="stat-card">
                    <div class="stat-label">${g.away_team} Edge</div>
                    <div class="stat-value" style="color:${edges.away.is_positive_ev ? 'var(--green)' : 'var(--red)'}; font-size:18px">${edges.away.edge_pct > 0 ? '+' : ''}${edges.away.edge_pct}%</div>
                    <div class="stat-sub">Market: ${edges.away.market_odds} | Kelly: ${edges.away.half_kelly_pct}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">${g.home_team} Edge</div>
                    <div class="stat-value" style="color:${edges.home.is_positive_ev ? 'var(--green)' : 'var(--red)'}; font-size:18px">${edges.home.edge_pct > 0 ? '+' : ''}${edges.home.edge_pct}%</div>
                    <div class="stat-sub">Market: ${edges.home.market_odds} | Kelly: ${edges.home.half_kelly_pct}%</div>
                </div>
                ${edges.total_line ? `
                <div class="stat-card">
                    <div class="stat-label">Total Line</div>
                    <div class="stat-value" style="font-size:18px; color:${edges.total_diff > 0.5 ? 'var(--green)' : edges.total_diff < -0.5 ? 'var(--red)' : 'var(--text-muted)'}">${edges.total_diff > 0 ? '+' : ''}${edges.total_diff}</div>
                    <div class="stat-sub">Line: ${edges.total_line} | Model: ${data.total_runs_mean.toFixed(1)}</div>
                </div>` : ''}
            </div>
        </div>` : ''}

        <!-- Charts -->
        <div class="sim-stats-grid">
            <div class="sim-panel">
                <h3>Run Distribution</h3>
                <div class="chart-container"><canvas id="runDistChart"></canvas></div>
            </div>
            <div class="sim-panel">
                <h3>Total Runs Distribution</h3>
                <div class="chart-container"><canvas id="totalRunsChart"></canvas></div>
            </div>
        </div>

        <!-- Player Projections -->
        <div class="sim-stats-grid">
            <div class="sim-panel">
                <h3>${g.away_team} Player Projections (DK Pts)</h3>
                ${renderPlayerProjections(data.away_projections, adj.away_batters)}
            </div>
            <div class="sim-panel">
                <h3>${g.home_team} Player Projections (DK Pts)</h3>
                ${renderPlayerProjections(data.home_projections, adj.home_batters)}
            </div>
        </div>

        <!-- Pitching -->
        <div class="sim-panel">
            <h3>Pitching Matchup</h3>
            <table>
                <thead><tr><th>Pitcher</th><th>Hand</th><th>ERA</th><th>WHIP</th><th>K/9</th><th>Workload</th></tr></thead>
                <tbody>
                    <tr>
                        <td><span class="player-name">${data.away_pitcher.name}</span> <span class="pos-badge">${g.away_team}</span></td>
                        <td class="num">${data.away_pitcher.hand || '?'}HP</td>
                        <td class="num">${data.away_pitcher.era}</td>
                        <td class="num">${data.away_pitcher.whip}</td>
                        <td class="num">${data.away_pitcher.k_per_9}</td>
                        <td>${workloadBadge(data.away_pitcher.workload)}</td>
                    </tr>
                    <tr>
                        <td><span class="player-name">${data.home_pitcher.name}</span> <span class="pos-badge">${g.home_team}</span></td>
                        <td class="num">${data.home_pitcher.hand || '?'}HP</td>
                        <td class="num">${data.home_pitcher.era}</td>
                        <td class="num">${data.home_pitcher.whip}</td>
                        <td class="num">${data.home_pitcher.k_per_9}</td>
                        <td>${workloadBadge(data.home_pitcher.workload)}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    renderRunDistChart(data, g);
    renderTotalRunsChart(data);
}

function renderPlayerProjections(projections, adjustments) {
    if (!projections || !projections.length) return '<p class="meta">No data</p>';
    const adjs = adjustments || [];
    return `<table class="proj-table">
        <thead><tr><th>#</th><th>Player</th><th>DK Med</th><th>Floor</th><th>Ceil</th><th>Avg H</th><th>Avg HR</th><th>RBI</th><th>Form</th></tr></thead>
        <tbody>
            ${projections.slice(0, 9).map((p, i) => {
                const adj = adjs[i] || {};
                return `<tr>
                    <td class="rank">${i + 1}</td>
                    <td class="player-name">${p.name} <span class="pos-badge">${adj.platoon || ''}</span></td>
                    <td class="num highlight">${p.dk_median.toFixed(1)}</td>
                    <td class="num" style="color:var(--text-faint)">${p.dk_p10.toFixed(1)}</td>
                    <td class="num" style="color:var(--green)">${p.dk_p90.toFixed(1)}</td>
                    <td class="num">${p.avg_hits.toFixed(2)}</td>
                    <td class="num">${p.avg_hr.toFixed(3)}</td>
                    <td class="num">${p.avg_rbi.toFixed(2)}</td>
                    <td>${formBadge(adj.form)}</td>
                </tr>`;
            }).join('')}
        </tbody>
    </table>`;
}

function renderRunDistChart(data, g) {
    const ctx = document.getElementById('runDistChart').getContext('2d');
    const maxRuns = Math.max(...Object.keys(data.away_runs_dist).map(Number), ...Object.keys(data.home_runs_dist).map(Number));
    const labels = Array.from({ length: Math.min(maxRuns + 1, 20) }, (_, i) => i);
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: g.away_team, data: labels.map(l => ((data.away_runs_dist[l] || 0) / data.n_sims * 100)), backgroundColor: 'rgba(59,130,246,0.6)', borderColor: 'rgba(59,130,246,1)', borderWidth: 1 },
                { label: g.home_team, data: labels.map(l => ((data.home_runs_dist[l] || 0) / data.n_sims * 100)), backgroundColor: 'rgba(34,197,94,0.6)', borderColor: 'rgba(34,197,94,1)', borderWidth: 1 }
            ]
        },
        options: chartOpts('Runs', '% of Sims')
    });
}

function renderTotalRunsChart(data) {
    const ctx = document.getElementById('totalRunsChart').getContext('2d');
    const labels = Object.keys(data.total_runs_dist).map(Number).sort((a, b) => a - b).filter(n => n <= 25);
    new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Total', data: labels.map(l => (data.total_runs_dist[l] / data.n_sims * 100)), backgroundColor: 'rgba(245,158,11,0.6)', borderColor: 'rgba(245,158,11,1)', borderWidth: 1 }] },
        options: { ...chartOpts('Total Runs', '% of Sims'), plugins: { legend: { display: false } } }
    });
}

function chartOpts(xLabel, yLabel) {
    return {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#8892a6', font: { size: 11, family: 'Inter' } } } },
        scales: {
            x: { title: { display: true, text: xLabel, color: '#5a6478', font: { size: 11 } }, ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: 'rgba(37,45,66,0.5)' } },
            y: { title: { display: true, text: yLabel, color: '#5a6478', font: { size: 11 } }, ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: 'rgba(37,45,66,0.5)' } }
        }
    };
}

// ─── DFS Projections ──────────────────────────────────────────
async function loadProjections() {
    const container = document.getElementById('projectionsTable');
    const btn = document.getElementById('loadProjections');
    btn.innerHTML = '<span class="spinner-inline"></span> Simulating...';
    btn.disabled = true;
    container.innerHTML = '<div class="loading">Simulating all games with adjustments... 30-60 sec.</div>';
    try {
        const res = await fetch(`${API}/api/projections`);
        const data = await res.json();
        projectionsData = data.projections;
        renderProjectionsTable(projectionsData);
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Error: ${err.message}</p></div>`;
    } finally {
        btn.innerHTML = 'Generate Projections';
        btn.disabled = false;
    }
}

function renderProjectionsTable(projections) {
    const container = document.getElementById('projectionsTable');
    if (!projections.length) { container.innerHTML = '<div class="empty-state">No data</div>'; return; }
    container.innerHTML = `
        <table id="projTable">
            <thead><tr>
                <th onclick="sortTable('rank')">#</th>
                <th onclick="sortTable('name')">Player</th>
                <th onclick="sortTable('team')">Team</th>
                <th onclick="sortTable('opp_pitcher')">vs Pitcher</th>
                <th>Hand</th>
                <th onclick="sortTable('dk_median')">DK Med</th>
                <th onclick="sortTable('dk_p10')">Floor</th>
                <th onclick="sortTable('dk_p75')">P75</th>
                <th onclick="sortTable('dk_p90')">Ceil</th>
                <th onclick="sortTable('dk_p99')">P99</th>
                <th onclick="sortTable('avg_hits')">Avg H</th>
                <th onclick="sortTable('avg_hr')">Avg HR</th>
                <th onclick="sortTable('avg_rbi')">RBI</th>
                <th>Park HR</th>
                <th onclick="sortTable('hit_rate')">Hit%</th>
                <th onclick="sortTable('hr_rate')">HR%</th>
            </tr></thead>
            <tbody>
                ${projections.map((p, i) => `<tr>
                    <td class="num" style="color:var(--text-faint)">${i + 1}</td>
                    <td class="player-name">${p.name}</td>
                    <td><span class="pos-badge">${p.team}</span></td>
                    <td style="font-size:12px;color:var(--text-muted)">${p.opp_pitcher}</td>
                    <td class="num">${p.opp_pitcher_hand || '?'}</td>
                    <td class="num highlight">${p.dk_median.toFixed(1)}</td>
                    <td class="num" style="color:var(--text-faint)">${p.dk_p10.toFixed(1)}</td>
                    <td class="num">${p.dk_p75.toFixed(1)}</td>
                    <td class="num" style="color:var(--green)">${p.dk_p90.toFixed(1)}</td>
                    <td class="num" style="color:var(--purple)">${p.dk_p99.toFixed(1)}</td>
                    <td class="num">${p.avg_hits.toFixed(2)}</td>
                    <td class="num">${p.avg_hr.toFixed(3)}</td>
                    <td class="num">${p.avg_rbi.toFixed(2)}</td>
                    <td class="num ${p.park_hr_factor > 1.05 ? 'positive' : p.park_hr_factor < 0.95 ? 'negative' : ''}">${p.park_hr_factor?.toFixed(2) || '1.00'}</td>
                    <td class="num">${(p.hit_rate * 100).toFixed(0)}%</td>
                    <td class="num">${(p.hr_rate * 100).toFixed(1)}%</td>
                </tr>`).join('')}
            </tbody>
        </table>`;
}

function sortTable(col) {
    if (currentSort.col === col) currentSort.dir = currentSort.dir === 'desc' ? 'asc' : 'desc';
    else currentSort = { col, dir: 'desc' };
    const sorted = [...projectionsData].sort((a, b) => {
        let aV = a[col], bV = b[col];
        if (typeof aV === 'string') return currentSort.dir === 'asc' ? aV.localeCompare(bV) : bV.localeCompare(aV);
        return currentSort.dir === 'asc' ? aV - bV : bV - aV;
    });
    renderProjectionsTable(sorted);
}

function filterProjections() {
    const pos = document.getElementById('posFilter').value;
    renderProjectionsTable(pos === 'all' ? projectionsData : projectionsData.filter(p => p.position === pos));
}

// ─── Betting Edges ────────────────────────────────────────────
async function loadEdges() {
    const container = document.getElementById('edgesTable');
    try {
        const res = await fetch(`${API}/api/edges`);
        const data = await res.json();
        renderEdgesTable(data);
    } catch (err) {
        container.innerHTML = `<div class="empty-state"><p>Error: ${err.message}</p></div>`;
    }
}

function renderEdgesTable(data) {
    const container = document.getElementById('edgesTable');
    const edges = data.edges || [];
    const hasSB = data.sportsbook_count > 0;

    container.innerHTML = `
        ${hasSB ? `<p class="meta" style="margin-bottom:12px">Consensus from ${data.sportsbook_count} sportsbook feeds. Green = +EV edge.</p>` :
            `<p class="meta" style="margin-bottom:12px">No sportsbook data loaded. Enter your <a href="https://the-odds-api.com" target="_blank" style="color:var(--accent)">Odds API</a> key above, or manually enter odds below.</p>`}
        <table>
            <thead><tr>
                <th>Matchup</th>
                <th>Away SP</th>
                <th>Home SP</th>
                <th>Model Away</th>
                <th>Model Home</th>
                <th>Fair Away</th>
                <th>Fair Home</th>
                ${hasSB ? '<th>Mkt Away</th><th>Mkt Home</th><th>Total</th><th>Edge</th>' :
                    '<th>Mkt Away</th><th>Mkt Home</th><th>Edge</th>'}
                <th>Park</th>
            </tr></thead>
            <tbody>
                ${edges.map(e => {
                    const awayEdge = e.away_edge;
                    const homeEdge = e.home_edge;
                    return `<tr class="edge-row">
                        <td><strong>${e.away_team} @ ${e.home_team}</strong></td>
                        <td style="font-size:12px">${e.away_pitcher}</td>
                        <td style="font-size:12px">${e.home_pitcher}</td>
                        <td class="num">${(e.model_away_prob * 100).toFixed(1)}%</td>
                        <td class="num">${(e.model_home_prob * 100).toFixed(1)}%</td>
                        <td class="num">${e.away_fair_odds}</td>
                        <td class="num">${e.home_fair_odds}</td>
                        ${hasSB && e.consensus_away_ml ? `
                            <td class="num book-odds">${e.consensus_away_ml > 0 ? '+' : ''}${e.consensus_away_ml}</td>
                            <td class="num book-odds">${e.consensus_home_ml > 0 ? '+' : ''}${e.consensus_home_ml}</td>
                            <td class="num book-odds">${e.consensus_total || '—'}</td>
                            <td>${renderEdge(awayEdge, homeEdge, e)}</td>
                        ` : `
                            <td><input type="text" class="edge-input" placeholder="+150" onchange="calcManualEdge('${e.game_id}','away',this.value,${e.model_away_prob})"></td>
                            <td><input type="text" class="edge-input" placeholder="-180" onchange="calcManualEdge('${e.game_id}','home',this.value,${e.model_home_prob})"></td>
                            <td class="num" id="edge-val-${e.game_id}">—</td>
                        `}
                        <td style="font-size:11px;color:var(--text-faint)">${e.park || ''}</td>
                    </tr>`;
                }).join('')}
            </tbody>
        </table>`;
}

function renderEdge(away, home, e) {
    if (!away) return '—';
    // Show the side with positive edge, or the smaller negative
    const best = away.edge_pct > home.edge_pct ? { side: e.away_team, ...away } : { side: e.home_team, ...home };
    const cls = best.is_positive_ev ? 'edge-positive' : 'edge-negative';
    let html = `<span class="edge-value ${cls}">${best.edge_pct > 0 ? '+' : ''}${best.edge_pct}% ${best.side}</span>`;
    if (best.is_positive_ev) html += `<br><span style="font-size:10px;color:var(--text-faint)">½K: ${best.half_kelly_pct}%</span>`;
    return html;
}

function calcManualEdge(gameId, side, oddsStr, modelProb) {
    const cell = document.getElementById(`edge-val-${gameId}`);
    const odds = parseInt(oddsStr.replace('+', ''));
    if (isNaN(odds)) { cell.innerHTML = '—'; return; }
    let implied = odds > 0 ? 100 / (odds + 100) : Math.abs(odds) / (Math.abs(odds) + 100);
    const edge = ((modelProb - implied) * 100).toFixed(1);
    const edgeNum = parseFloat(edge);
    if (edgeNum > 0) {
        const b = odds > 0 ? odds / 100 : 100 / Math.abs(odds);
        const kelly = ((modelProb * b - (1 - modelProb)) / b * 100).toFixed(1);
        cell.innerHTML = `<span class="edge-value edge-positive">+${edge}%</span><br><span style="font-size:10px;color:var(--text-faint)">Kelly: ${kelly}%</span>`;
    } else {
        cell.innerHTML = `<span class="edge-value edge-negative">${edge}%</span>`;
    }
}
