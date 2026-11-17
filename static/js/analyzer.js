// Single Text Sentiment Analyzer Client Handler

document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('sentimentForm');
  const textarea = document.getElementById('inputText');
  const modelSelect = document.getElementById('modelSelect');
  const charCounter = document.getElementById('charCounter');
  const alertBox = document.getElementById('analyzerAlert');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const resultCard = document.getElementById('resultCard');

  if (!form || !textarea) return;

  // Live character counter
  textarea.addEventListener('input', function () {
    const len = textarea.value.length;
    charCounter.innerText = `${len.toLocaleString()} / 5,000 characters`;
    if (len > 5000) {
      charCounter.style.color = 'var(--negative)';
    } else {
      charCounter.style.color = 'var(--text-muted)';
    }
  });

  // Sample prompt buttons
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      textarea.value = this.getAttribute('data-sample');
      textarea.dispatchEvent(new Event('input'));
      alertBox.style.display = 'none';
    });
  });

  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      textarea.value = '';
      textarea.dispatchEvent(new Event('input'));
      resultCard.style.display = 'none';
      alertBox.style.display = 'none';
    });
  }

  // Handle Form Submission
  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    alertBox.style.display = 'none';

    const text = textarea.value.trim();
    if (!text) {
      alertBox.innerText = 'Please enter text to analyze.';
      alertBox.style.display = 'block';
      return;
    }

    const modelName = modelSelect.value;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Sentiment...';

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          model_name: modelName
        })
      });

      const data = await response.json();

      if (!response.ok || data.status === 'error') {
        alertBox.innerText = data.message || 'Analysis request failed.';
        alertBox.style.display = 'block';
        return;
      }

      // Populate Result Card
      resultCard.style.display = 'block';

      const sentimentEl = document.getElementById('resultSentiment');
      let icon = '<i class="fa-solid fa-face-meh"></i>';
      let color = 'var(--neutral)';

      if (data.sentiment === 'Positive') {
        icon = '<i class="fa-solid fa-face-smile"></i>';
        color = 'var(--positive)';
      } else if (data.sentiment === 'Negative') {
        icon = '<i class="fa-solid fa-face-frown"></i>';
        color = 'var(--negative)';
      }

      sentimentEl.innerHTML = `<span style="color:${color};">${icon} ${data.sentiment}</span>`;
      document.getElementById('resultConfidence').innerText = `${data.confidence_percent}%`;

      // Update Probability Bars
      const posPct = (data.positive_score * 100).toFixed(1);
      const negPct = (data.negative_score * 100).toFixed(1);
      const neuPct = (data.neutral_score * 100).toFixed(1);

      document.getElementById('posScoreText').innerText = `${posPct}%`;
      document.getElementById('posScoreBar').style.width = `${posPct}%`;

      document.getElementById('negScoreText').innerText = `${negPct}%`;
      document.getElementById('negScoreBar').style.width = `${negPct}%`;

      document.getElementById('neuScoreText').innerText = `${neuPct}%`;
      document.getElementById('neuScoreBar').style.width = `${neuPct}%`;

      document.getElementById('modelUsedBadge').innerText = `Model: ${data.model_name}`;

      resultCard.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
      alertBox.innerText = 'Network error occurred: ' + err.message;
      alertBox.style.display = 'block';
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart"></i> Analyze Sentiment';
    }
  });
});
