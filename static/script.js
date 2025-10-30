let lastResponseTime = Date.now();
let avgResponse = 0;
let frameCount = 0;

async function updateStatus() {
  const t0 = performance.now();
  const res = await fetch('/status');
  const j = await res.json();
  const t1 = performance.now();

  // Calculate response time (ms)
  const responseTime = Math.round(t1 - t0);
  avgResponse = (avgResponse * frameCount + responseTime) / (frameCount + 1);
  frameCount++;

  // Update distance, volume, calibration
  document.getElementById('dist').textContent = Math.round(j.distance);
  document.getElementById('vol').textContent = j.volume + '%';
  document.getElementById('fill').style.width = j.volume + '%';
  document.getElementById('calMin').textContent = Math.round(j.calib_min);
  document.getElementById('calMax').textContent = Math.round(j.calib_max);
  document.getElementById('calApplied').textContent = j.calib_applied;

  // Gesture highlights
  const gestures = ['gOpen', 'gClosed', 'gPinch'];
  gestures.forEach(id => document.getElementById(id).classList.remove('active'));
  if (j.gesture === 'OPEN') document.getElementById('gOpen').classList.add('active');
  if (j.gesture === 'CLOSED') document.getElementById('gClosed').classList.add('active');
  if (j.gesture === 'PINCH') document.getElementById('gPinch').classList.add('active');

  // ======== SYSTEM METRICS UPDATE ========
  // Accuracy estimation based on gesture stability
  const acc = j.distance > 0 ? Math.min(100, Math.abs(100 - (Math.abs(j.distance - 100) / 100) * 100)) : 0;

  // Update text values
  updateGauge('respGauge', responseTime, 500); // expected max 500ms
  updateGauge('volGauge', j.volume, 100);
  updateGauge('accGauge', acc, 100);
  updateGauge('distGauge', j.distance, 200);

  document.getElementById('resp').textContent = responseTime + 'ms';
  document.getElementById('volPerc').textContent = j.volume + '%';
  document.getElementById('acc').textContent = Math.round(acc) + '%';
  document.getElementById('distVal').textContent = Math.round(j.distance);
}

function updateGauge(id, value, maxValue) {
  const elem = document.getElementById(id);
  const circle = elem.querySelector('.meter');
  const maxCircum = 251.2;
  const pct = Math.min(1, value / maxValue);
  const offset = maxCircum * (1 - pct);
  if (circle) circle.style.strokeDashoffset = offset;

  // Glow & color logic
  elem.classList.remove('good', 'medium', 'bad');
  if (pct >= 0.75) elem.classList.add('good');
  else if (pct >= 0.4) elem.classList.add('medium');
  else elem.classList.add('bad');
}


function refreshPlot() {
  document.getElementById('plot').src = '/plot.png?ts=' + Date.now();
}

// Manual calibration
async function applyManual() {
  const min = parseFloat(document.getElementById('minInput').value);
  const max = parseFloat(document.getElementById('maxInput').value);
  if (isNaN(min) || isNaN(max) || max <= min) {
    alert("⚠️ Please enter valid Min < Max values.");
    return;
  }
  await fetch('/calib/apply_manual', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({min, max})
  });
  alert("✅ Calibration applied successfully!");
  updateStatus();
}

// Camera controls
async function startCam() { await fetch('/camera/start', {method:'POST'}); }
async function pauseCam() { await fetch('/camera/pause', {method:'POST'}); }
async function stopCam() { await fetch('/camera/stop', {method:'POST'}); }

window.addEventListener('load', () => {
  document.getElementById('video').src = '/video_feed';
  document.getElementById('applyManual').onclick = applyManual;
  document.getElementById('startCam').onclick = startCam;
  document.getElementById('pauseCam').onclick = pauseCam;
  document.getElementById('stopCam').onclick = stopCam;


  setInterval(updateStatus, 400);   // was 200ms — reduced for CPU efficiency
  setInterval(refreshPlot, 1500);   // was 1000ms
});
function updateGauge(id, percent, color) {
  const circle = document.querySelector(`#${id} .meter`);
  const offset = 251.2 - (percent / 100) * 251.2;
  circle.style.strokeDashoffset = offset;
  circle.style.stroke = color;
}

