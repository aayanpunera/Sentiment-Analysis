// Chart.js visualizations for Sentiment Dashboard

function initDashboardCharts(data) {
  const donutCtx = document.getElementById('sentimentDonutChart');
  const trendCtx = document.getElementById('sentimentTrendChart');

  const total = data.positive + data.negative + data.neutral;

  // Donut Chart
  if (donutCtx) {
    if (total === 0) {
      // Empty state donut placeholder
      new Chart(donutCtx, {
        type: 'doughnut',
        data: {
          labels: ['No Data Yet'],
          datasets: [{
            data: [1],
            backgroundColor: ['#E5E7EB']
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom' },
            tooltip: { enabled: false }
          }
        }
      });
    } else {
      new Chart(donutCtx, {
        type: 'doughnut',
        data: {
          labels: ['Positive', 'Negative', 'Neutral'],
          datasets: [{
            data: [data.positive, data.negative, data.neutral],
            backgroundColor: ['#10B981', '#EF4444', '#3B82F6'],
            borderWidth: 2,
            borderColor: '#FFFFFF',
            hoverOffset: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '70%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                boxWidth: 12,
                padding: 16,
                font: { family: 'Inter', size: 12, weight: '500' }
              }
            }
          }
        }
      });
    }
  }

  // Sentiment Trend Chart (Line / Grouped Bar)
  if (trendCtx) {
    const trends = data.trends || [];
    if (trends.length === 0) {
      // Empty state
      new Chart(trendCtx, {
        type: 'bar',
        data: {
          labels: ['Today'],
          datasets: [{
            label: 'No Activity Yet',
            data: [0],
            backgroundColor: '#E5E7EB'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, max: 5 }
          }
        }
      });
    } else {
      const labels = trends.map(t => t.date);
      const posData = trends.map(t => t.positive);
      const negData = trends.map(t => t.negative);
      const neuData = trends.map(t => t.neutral);

      new Chart(trendCtx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Positive',
              data: posData,
              backgroundColor: '#10B981',
              borderRadius: 4
            },
            {
              label: 'Negative',
              data: negData,
              backgroundColor: '#EF4444',
              borderRadius: 4
            },
            {
              label: 'Neutral',
              data: neuData,
              backgroundColor: '#3B82F6',
              borderRadius: 4
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              grid: { display: false }
            },
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: '#F3F4F6' }
            }
          },
          plugins: {
            legend: {
              position: 'bottom',
              labels: {
                boxWidth: 12,
                padding: 16,
                font: { family: 'Inter', size: 12 }
              }
            }
          }
        }
      });
    }
  }
}
