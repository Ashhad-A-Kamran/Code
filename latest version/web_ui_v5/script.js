const API_BASE = "http://127.0.0.1:5000/api";
let activeModelId = null;
let charts = {};
let historyData = {}; // model_id -> array of log objects
let baselineMetrics = null; // Stores {acc, fair, nrg} when parameters are applied

// Initialize Charts
function initCharts() {
    Chart.defaults.color = '#8b949e';
    Chart.defaults.font.family = 'ui-monospace, monospace';

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { color: '#30363d' } },
            y: { grid: { color: '#30363d' } }
        }
    };

    charts['loss'] = new Chart(document.getElementById('chartLoss'), {
        type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#58a6ff', tension: 0.1, borderWidth: 2 }] },
        options: commonOptions
    });
    charts['fairness'] = new Chart(document.getElementById('chartFairness'), {
        type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#3fb950', tension: 0.1, borderWidth: 2 }] },
        options: commonOptions
    });
    charts['bias'] = new Chart(document.getElementById('chartBias'), {
        type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#f85149', tension: 0.1, borderWidth: 2 }] },
        options: commonOptions
    });
    charts['energy'] = new Chart(document.getElementById('chartEnergy'), {
        type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: '#d29922', tension: 0.1, borderWidth: 2 }] },
        options: commonOptions
    });
}

function showToast(msg) {
    const t = document.getElementById('toast');
    t.innerText = msg;
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 3000);
}

// Sliders UI update
function updateSliderVal(type) {
    document.getElementById(`val_${type}`).innerText = parseFloat(document.getElementById(`w_${type}`).value).toFixed(1);
}

// Handle UI changes when model type is selected
function handleModelChange() {
    const type = document.getElementById('new_model_type').value;
    const container = document.getElementById('dataset_container');
    if (type === 'resnet') {
        container.style.display = 'none';
    } else {
        container.style.display = 'block';
    }
}

async function createModel() {
    const name = document.getElementById('new_model_name').value.trim() || `Node_${Date.now()}`;
    const epochs = document.getElementById('new_epochs').value;
    const type = document.getElementById('new_model_type').value;
    
    // If ResNet is selected, use a fixed mode for the dataset
    let dataset = document.getElementById('new_dataset').value;
    if (type === 'resnet') {
        dataset = 'synthetic_vision'; 
    }
    
    const fairnessMethod = document.getElementById('new_fairness_method').value;

    // Warn user about slow methods
    if (fairnessMethod === 'nsga2') {
        showToast('⚠️ NSGA-II selected: fairness will update every 10 epochs (computationally heavy)');
    } else if (fairnessMethod === 'roc') {
        showToast('ℹ️ ROC selected: fairness will update every 5 epochs');
    }

    try {
        const res = await fetch(`${API_BASE}/create/${name}/${type}/${epochs}/${dataset}/${fairnessMethod}`, { method: 'POST' });
        const data = await res.json();
        if (data.status === "started") {
            showToast(`Instance ${name} initialized [${dataset} / ${fairnessMethod.toUpperCase()}]`);
            historyData[name] = [];
            selectModel(name);
            pollModels();
        } else {
            alert("ID Collision or Creation Failed");
        }
    } catch (e) {
        alert("Failed to connect to Controller");
    }
}

async function sendCommand(cmd, args = "{}") {
    if (!activeModelId) return;
    try {
        await fetch(`${API_BASE}/command/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: activeModelId, command: cmd, args: args })
        });
        if (cmd === 'update_weights') showToast("Parameters Synced");
    } catch (e) { console.error("Command failed", e); }
}

function applyWeights() {
    const acc = parseFloat(document.getElementById('w_acc').value);
    const fair = parseFloat(document.getElementById('w_fair').value);
    const nrg = parseFloat(document.getElementById('w_nrg').value);
    sendCommand('update_weights', JSON.stringify({ accuracy: acc, fairness: fair, energy: nrg }));

    // Set baseline metrics to calculate "Change Since Tuning"
    if (activeModelId && historyData[activeModelId] && historyData[activeModelId].length > 0) {
        const hist = historyData[activeModelId];
        const latest = hist[hist.length - 1];
        baselineMetrics = {
            acc: latest.accuracy || 0,
            fair: latest.fairness || 0,
            nrg: latest.energy_consumed || 0
        };
        showToast("Parameters Synced. Baseline Established.");
    } else {
        showToast("Parameters Synced");
    }
}

function downloadReport() {
    if (!activeModelId || !historyData[activeModelId] || historyData[activeModelId].length === 0) {
        showToast("No data to download.");
        return;
    }
    
    const rows = [
        ["Epoch", "Accuracy", "Fairness (DPD)", "Bias (Mean Diff)", "Loss", "Energy (kWh)", "Power Draw (W)"]
    ];
    
    historyData[activeModelId].forEach(d => {
        rows.push([
            d.epoch, 
            d.accuracy, 
            d.fairness, 
            d.bias, 
            d.loss, 
            d.energy_consumed, 
            d.power_draw
        ]);
    });
    
    const csvContent = "data:text/csv;charset=utf-8," + rows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `telemetry_report_${activeModelId}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function deleteCurrentModel() {
    if (!activeModelId) return;
    if (!confirm(`Destroy instance ${activeModelId}? Data will be lost.`)) return;

    try {
        await fetch(`${API_BASE}/delete/${activeModelId}`, { method: 'DELETE' });
        delete historyData[activeModelId];
        activeModelId = null;
        showToast("Instance Destroyed");
        document.getElementById('main_workspace').style.display = 'block';
        document.getElementById('dashboard').style.display = 'none';
        document.getElementById('empty_state').style.display = 'block';
        document.getElementById('right_panel').style.display = 'none';
        pollModels();
    } catch (e) { }
}

function selectModel(mid) {
    activeModelId = mid;
    document.getElementById('empty_state').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('right_panel').style.display = 'block';
    document.getElementById('view_model_id').innerText = `Node: ${mid}`;

    // Reset charts if needed
    if (!historyData[mid]) historyData[mid] = [];
    updateCharts(mid);
    pollModels(); // instantly trigger refresh
}

function updateCharts(mid) {
    if (!historyData[mid] || historyData[mid].length === 0) {
        Object.values(charts).forEach(c => {
            c.data.labels = [];
            c.data.datasets[0].data = [];
            c.update();
        });
        return;
    }
    const data = historyData[mid];
    const labels = data.map(d => d.epoch);

    charts['loss'].data.labels = labels;
    charts['loss'].data.datasets[0].data = data.map(d => d.loss);
    charts['loss'].update();

    charts['fairness'].data.labels = labels;
    charts['fairness'].data.datasets[0].data = data.map(d => d.fairness);
    charts['fairness'].update();

    charts['bias'].data.labels = labels;
    charts['bias'].data.datasets[0].data = data.map(d => d.bias);
    charts['bias'].update();

    charts['energy'].data.labels = labels;
    charts['energy'].data.datasets[0].data = data.map(d => d.energy_consumed);
    charts['energy'].update();
}

function renderReportTable(mid) {
    const tbody = document.getElementById('report_table_body');
    tbody.innerHTML = '';
    if (!historyData[mid]) return;
    
    // Reverse to show most recent epochs at the top
    const data = [...historyData[mid]].reverse();
    
    data.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="padding: 8px; border-bottom: 1px solid var(--border);">${d.epoch}</td>
            <td style="padding: 8px; border-bottom: 1px solid var(--border);">${d.loss ? d.loss.toFixed(4) : '0.00'}</td>
            <td style="padding: 8px; border-bottom: 1px solid var(--border);">${d.accuracy ? (d.accuracy * 100).toFixed(1) + '%' : '0.0%'}</td>
            <td style="padding: 8px; border-bottom: 1px solid var(--border);">${d.fairness ? d.fairness.toFixed(4) : '0.00'}</td>
            <td style="padding: 8px; border-bottom: 1px solid var(--border);">${d.bias ? d.bias.toFixed(4) : '0.00'}</td>
            <td style="padding: 8px; border-bottom: 1px solid var(--border); color: var(--warning);">${d.energy_consumed ? d.energy_consumed.toFixed(6) : '0.000000'}</td>
        `;
        tbody.appendChild(tr);
    });
}

async function pollModels() {
    try {
        const res = await fetch(`${API_BASE}/models?_t=${Date.now()}`);
        const models = await res.json();

        // Update Sidebar List
        const listContainer = document.getElementById('model_list');
        listContainer.innerHTML = '';

        for (const [mid, info] of Object.entries(models)) {
            const div = document.createElement('div');
            div.className = `model-card ${mid === activeModelId ? 'active' : ''}`;
            div.onclick = () => selectModel(mid);
            const datasetLabel = (info.dataset || 'adult').replace('_', ' ');
            const methodLabel = (info.fairness_method || 'dpd').toUpperCase();
            div.innerHTML = `
                <span class="mc-name">${mid}</span>
                <span class="mc-status ${info.status}">${info.status.toUpperCase()}</span>
                <span style="font-size:0.65rem; color: var(--text-muted); margin-top: 2px;">${datasetLabel} · ${methodLabel}</span>
            `;
            listContainer.appendChild(div);
        }

        // Update Active Dashboard
        if (activeModelId && models[activeModelId]) {
            const statusInfo = models[activeModelId];

            // Poll Log
            const logRes = await fetch(`${API_BASE}/logs/${activeModelId}?_t=${Date.now()}`);
            const log = await logRes.json();

            document.getElementById('kpi_status').innerText = statusInfo.status.toUpperCase();
            document.getElementById('kpi_epoch').innerText = `${log.epoch || 0}/${log.total_epochs || statusInfo.total}`;
            document.getElementById('kpi_acc').innerText = log.accuracy ? `${(log.accuracy * 100).toFixed(1)}%` : '0.0%';
            document.getElementById('kpi_nrg').innerText = log.energy_consumed ? log.energy_consumed.toFixed(6) : '0.000000';
            
            // New KPIs
            document.getElementById('kpi_time').innerText = log.elapsed_time ? `${log.elapsed_time.toFixed(1)}s` : '0.0s';
            document.getElementById('kpi_iter').innerText = `${log.iteration || 0}/${log.total_iterations || 0}`;
            
            if (baselineMetrics) {
                const dAcc = (log.accuracy || 0) - baselineMetrics.acc;
                const dFair = (log.fairness || 0) - baselineMetrics.fair; // Note: lower DPD (fairness) is better
                const dNrg = (log.energy_consumed || 0) - baselineMetrics.nrg;
                
                document.getElementById('kpi_delta_acc').innerText = `${(dAcc * 100).toFixed(1)}%`;
                document.getElementById('kpi_delta_fair').innerText = `${dFair > 0 ? '+' : ''}${dFair.toFixed(4)}`;
                document.getElementById('kpi_delta_nrg').innerText = `${dNrg > 0 ? '+' : ''}${dNrg.toFixed(6)}`;
            } else {
                document.getElementById('kpi_delta_acc').innerText = '--';
                document.getElementById('kpi_delta_fair').innerText = '--';
                document.getElementById('kpi_delta_nrg').innerText = '--';
            }

            // Update History Buffer
            if (log && log.epoch) {
                if (!historyData[activeModelId]) historyData[activeModelId] = [];
                const hist = historyData[activeModelId];
                
                if (hist.length === 0 || hist[hist.length - 1].epoch !== log.epoch) {
                    hist.push(log);
                } else {
                    hist[hist.length - 1] = log; // Update latest batch inside the same epoch
                }
                updateCharts(activeModelId);
                renderReportTable(activeModelId);
            }
        }
    } catch (e) { }
}

// Main Loop
window.onload = () => {
    initCharts();
    pollModels();
    setInterval(pollModels, 1000);
};
