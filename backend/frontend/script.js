// 🔥 IMPORTANT CHANGE: Same-origin API (NO localhost hardcoding)
const API_URL = '/api';

// Load data on page load
document.addEventListener('DOMContentLoaded', () => {
    loadAnalytics();
    loadEquipmentList();
});

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await makePrediction();
});

// Make prediction
async function makePrediction() {
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing...';

    const formData = {
        air_temperature: parseFloat(document.getElementById('airTemp').value),
        process_temperature: parseFloat(document.getElementById('processTemp').value),
        rotational_speed: parseInt(document.getElementById('rotSpeed').value),
        torque: parseFloat(document.getElementById('torque').value),
        tool_wear: parseInt(document.getElementById('toolWear').value),
        type: document.getElementById('machineType').value
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please ensure server is running.');
    } finally {
        btn.disabled = false;
        btn.textContent = '🔍 Predict Equipment Health';
    }
}

// Display prediction results
function displayResults(result) {
    document.getElementById('resultsContainer').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';

    // Status badge
    const statusDiv = document.getElementById('resultStatus');
    statusDiv.textContent = result.status.toUpperCase();
    statusDiv.className = `result-status status-${result.status}`;

    // Risk score circle
    const riskScore = document.getElementById('riskScore');
    const scoreValue = riskScore.querySelector('.score-value');
    scoreValue.textContent = `${result.risk_score.toFixed(1)}%`;

    riskScore.className = 'risk-score-circle';
    if (result.risk_score < 30) {
        riskScore.classList.add('risk-low');
    } else if (result.risk_score < 60) {
        riskScore.classList.add('risk-medium');
    } else {
        riskScore.classList.add('risk-high');
    }

    // Probabilities
    document.getElementById('healthyProb').textContent =
        `${(result.healthy_probability * 100).toFixed(1)}%`;
    document.getElementById('failureProb').textContent =
        `${(result.failure_probability * 100).toFixed(1)}%`;

    // Recommendation
    const recDiv = document.getElementById('recommendation');
    recDiv.innerHTML = `
        <h4>💡 Recommendation</h4>
        <p>${result.recommendation}</p>
    `;
}

// Load analytics
async function loadAnalytics() {
    try {
        const response = await fetch(`${API_URL}/analytics`);
        if (!response.ok) throw new Error('Failed to load analytics');

        const data = await response.json();

        document.getElementById('totalMachines').textContent =
            data.statistics.total_machines;
        document.getElementById('failureRate').textContent =
            `${data.statistics.failure_rate.toFixed(1)}%`;
        document.getElementById('avgTemp').textContent =
            `${data.statistics.avg_temperature.toFixed(1)}K`;
        document.getElementById('highRisk').textContent =
            data.statistics.high_risk_machines;

        drawFailureChart(data);
    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

// Load equipment list
async function loadEquipmentList() {
    try {
        const response = await fetch(`${API_URL}/equipment-list`);
        if (!response.ok) throw new Error('Failed to load equipment');

        const equipment = await response.json();
        displayEquipmentTable(equipment);
    } catch (error) {
        console.error('Error loading equipment:', error);
        document.getElementById('equipmentTableBody').innerHTML =
            '<tr><td colspan="9" class="loading">Error loading equipment data</td></tr>';
    }
}

// Display equipment table
function displayEquipmentTable(equipment) {
    const tbody = document.getElementById('equipmentTableBody');
    tbody.innerHTML = '';

    equipment.slice(0, 15).forEach(machine => {
        const row = document.createElement('tr');

        let statusClass = 'healthy';
        if (machine.risk_score > 60) statusClass = 'critical';
        else if (machine.risk_score > 30) statusClass = 'warning';

        row.innerHTML = `
            <td><strong>${machine.product_id}</strong></td>
            <td>${machine.type}</td>
            <td>${machine.air_temperature.toFixed(1)}K</td>
            <td>${machine.process_temperature.toFixed(1)}K</td>
            <td>${machine.rotational_speed}</td>
            <td>${machine.torque.toFixed(1)}Nm</td>
            <td>${machine.tool_wear}min</td>
            <td>${machine.risk_score.toFixed(1)}%</td>
            <td><span class="status-badge badge-${statusClass}">
                ${machine.status.toUpperCase()}</span></td>
        `;

        tbody.appendChild(row);
    });
}

// Draw simple failure chart
function drawFailureChart(data) {
    const canvas = document.getElementById('failureChart');
    const ctx = canvas.getContext('2d');

    canvas.width = canvas.offsetWidth;
    canvas.height = 300;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 50;

    ctx.clearRect(0, 0, width, height);

    const failureTypes = data.failure_types;
    const labels = Object.keys(failureTypes);
    const values = Object.values(failureTypes);
    const maxValue = Math.max(...values);

    const barWidth = (width - padding * 2) / labels.length - 20;
    const scale = (height - padding * 2) / maxValue;

    labels.forEach((label, i) => {
        const x = padding + i * (barWidth + 20);
        const barHeight = values[i] * scale;
        const y = height - padding - barHeight;

        const gradient = ctx.createLinearGradient(0, y, 0, height - padding);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');

        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);

        ctx.fillStyle = '#333';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(values[i], x + barWidth / 2, y - 10);

        ctx.fillStyle = '#666';
        ctx.font = '12px Arial';
        ctx.fillText(label, x + barWidth / 2, height - padding + 20);
    });
}
