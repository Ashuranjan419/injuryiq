// Prediction Form JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultsContainer = document.getElementById('resultsContainer');
    const resultsDiv = document.getElementById('results');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Get form data
        const formData = {
            player_name: document.getElementById('player_name').value,
            age: parseInt(document.getElementById('age').value),
            position: document.getElementById('position').value,
            injury_type: document.getElementById('injury_type').value,
            injury_severity: document.getElementById('injury_severity').value,
            previous_injuries: parseInt(document.getElementById('previous_injuries').value),
            fitness_level: parseInt(document.getElementById('fitness_level').value),
            model_type: document.getElementById('model_type').value,
            save_tracking: document.getElementById('save_tracking') ? document.getElementById('save_tracking').checked : false,
            notes: document.getElementById('notes') ? document.getElementById('notes').value : ''
        };

        // Validate form
        if (!validateForm(formData)) {
            return;
        }

        // Get submit button
        const submitButton = form.querySelector('button[type="submit"]');
        const originalText = submitButton.innerHTML;

        // Show loading state
        showLoading(submitButton);

        try {
            // Make API request
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
                resultsContainer.style.display = 'block';
                
                // Show tracking notification if saved
                if (data.tracking_saved) {
                    showNotification('✅ Player added to tracking!', 'success');
                }
                
                // Smooth scroll to results
                resultsContainer.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });
            } else {
                showError(data.error || 'An error occurred during prediction');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Failed to connect to the server. Please try again.');
        } finally {
            // Hide loading state
            hideLoading(submitButton, originalText);
        }
    });
});

function validateForm(data) {
    // Check age range
    if (data.age < 18 || data.age > 45) {
        alert('Age must be between 18 and 45');
        return false;
    }

    // Check fitness level range
    if (data.fitness_level < 1 || data.fitness_level > 10) {
        alert('Fitness level must be between 1 and 10');
        return false;
    }

    // Check previous injuries
    if (data.previous_injuries < 0) {
        alert('Previous injuries cannot be negative');
        return false;
    }

    return true;
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const resultsContainer = document.getElementById('resultsContainer');
    const predictGrid = document.querySelector('.predict-grid');
    
    // Add class to expand grid
    if (predictGrid) {
        predictGrid.classList.add('has-results');
    }
    
    // Determine risk class for styling
    let riskClass = 'risk-low';
    if (data.risk_level === 'Medium') {
        riskClass = 'risk-medium';
    } else if (data.risk_level === 'High') {
        riskClass = 'risk-high';
    }

    // Build results HTML
    const resultsHTML = `
        <div class="result-card">
            <div class="result-header">
                ${data.player_name}
            </div>
            
            <div class="result-item">
                <span class="result-label">Model Used:</span>
                <span class="result-value">${data.model_used}</span>
            </div>
            
            <div class="result-item">
                <span class="result-label">Position:</span>
                <span class="result-value">${data.input_data.position}</span>
            </div>
            
            <div class="result-item">
                <span class="result-label">Injury Type:</span>
                <span class="result-value">${data.input_data.injury_type}</span>
            </div>
            
            <div class="result-item">
                <span class="result-label">Injury Severity:</span>
                <span class="result-value">${data.input_data.injury_severity}</span>
            </div>
        </div>

        <div class="result-card" style="margin-top: 1rem;">
            <div class="result-header" style="color: var(--primary-color);">
                Predicted Recovery Time
            </div>
            
            <div class="result-item">
                <span class="result-label">Recovery Days:</span>
                <span class="result-value" style="font-size: 1.5rem; color: var(--primary-color);">
                    ${data.recovery_days} days
                </span>
            </div>
            
            <div class="result-item">
                <span class="result-label">Recovery Weeks:</span>
                <span class="result-value" style="font-size: 1.5rem; color: var(--primary-color);">
                    ${data.recovery_weeks} weeks
                </span>
            </div>
        </div>

        <div class="result-card" style="margin-top: 1rem;">
            <div class="result-header" style="color: var(--${data.risk_color}-color);">
                Setback Risk Assessment
            </div>
            
            <div class="result-item">
                <span class="result-label">Setback Probability:</span>
                <span class="result-value">${data.setback_probability}%</span>
            </div>
            
            <div class="result-item">
                <span class="result-label">Risk Level:</span>
                <span class="risk-badge ${riskClass}">${data.risk_level}</span>
            </div>
        </div>

        <div style="margin-top: 1.5rem; padding: 1rem; background: #eff6ff; border-radius: 8px; border-left: 4px solid var(--primary-color);">
            <strong>Recommendation:</strong>
            ${getRecommendation(data)}
        </div>
    `;

    resultsDiv.innerHTML = resultsHTML;
}

function getRecommendation(data) {
    let recommendation = '';
    
    if (data.risk_level === 'Low') {
        recommendation = `The player is expected to recover in approximately ${data.recovery_days} days with a low risk of setbacks. Monitor progress regularly and maintain the current rehabilitation plan.`;
    } else if (data.risk_level === 'Medium') {
        recommendation = `The player is expected to recover in approximately ${data.recovery_days} days but shows a moderate risk of setbacks. Consider implementing additional preventive measures and closer monitoring during rehabilitation.`;
    } else {
        recommendation = `The player is expected to recover in approximately ${data.recovery_days} days but has a high risk of setbacks. It is strongly recommended to adopt a conservative rehabilitation approach with frequent medical evaluations.`;
    }
    
    return recommendation;
}

function showError(message) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="alert alert-danger">
            <strong>Error:</strong> ${message}
        </div>
    `;
    document.getElementById('resultsContainer').style.display = 'block';
}

function showLoading(button) {
    button.disabled = true;
    button.innerHTML = '<span class="loading"></span> Predicting...';
}

function hideLoading(button, text) {
    button.disabled = false;
    button.innerHTML = text;
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
