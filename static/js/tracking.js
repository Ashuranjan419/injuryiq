/**
 * Player Tracking JavaScript
 * Handles recovery countdown, player management, and updates
 */

let allPlayers = [];
let updateInterval = null;

// Initialize tracking page
document.addEventListener('DOMContentLoaded', () => {
    loadTrackedPlayers();
    
    // Set up filter listeners
    document.getElementById('statusFilter').addEventListener('change', filterAndDisplayPlayers);
    document.getElementById('sortBy').addEventListener('change', filterAndDisplayPlayers);
    
    // Update countdown every minute
    updateInterval = setInterval(updateCountdowns, 60000);
});

// Load tracked players from API
async function loadTrackedPlayers() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    const emptyState = document.getElementById('emptyState');
    const playersGrid = document.getElementById('playersGrid');
    
    try {
        const response = await fetch('/api/tracking');
        const data = await response.json();
        
        if (data.success) {
            allPlayers = data.players;
            
            loadingIndicator.style.display = 'none';
            
            if (allPlayers.length === 0) {
                emptyState.style.display = 'block';
                playersGrid.style.display = 'none';
            } else {
                emptyState.style.display = 'none';
                playersGrid.style.display = 'grid';
                filterAndDisplayPlayers();
            }
        } else {
            showError('Failed to load tracked players');
        }
    } catch (error) {
        console.error('Error loading tracked players:', error);
        showError('An error occurred while loading players');
        loadingIndicator.style.display = 'none';
    }
}

// Filter and display players based on current filters
function filterAndDisplayPlayers() {
    const statusFilter = document.getElementById('statusFilter').value;
    const sortBy = document.getElementById('sortBy').value;
    
    let filteredPlayers = [...allPlayers];
    
    // Apply status filter
    if (statusFilter !== 'all') {
        filteredPlayers = filteredPlayers.filter(player => {
            switch (statusFilter) {
                case 'active':
                    return !player.is_recovered && player.days_remaining > 3;
                case 'almost':
                    return !player.is_recovered && player.days_remaining > 0 && player.days_remaining <= 3;
                case 'completed':
                    return !player.is_recovered && player.days_remaining === 0;
                case 'recovered':
                    return player.is_recovered;
                default:
                    return true;
            }
        });
    }
    
    // Apply sorting
    filteredPlayers.sort((a, b) => {
        switch (sortBy) {
            case 'days_asc':
                return a.days_remaining - b.days_remaining;
            case 'days_desc':
                return b.days_remaining - a.days_remaining;
            case 'recent':
                return new Date(b.prediction_date) - new Date(a.prediction_date);
            case 'oldest':
                return new Date(a.prediction_date) - new Date(b.prediction_date);
            default:
                return 0;
        }
    });
    
    displayPlayers(filteredPlayers);
}

// Display players in the grid
function displayPlayers(players) {
    const playersGrid = document.getElementById('playersGrid');
    
    if (players.length === 0) {
        playersGrid.innerHTML = '<div class="no-results"><p>No players match the selected filters</p></div>';
        return;
    }
    
    playersGrid.innerHTML = players.map(player => createPlayerCard(player)).join('');
    
    // Add event listeners after DOM elements are created
    players.forEach(player => {
        const deleteBtn = document.getElementById(`delete-${player.id}`);
        const recoverBtn = document.getElementById(`recover-${player.id}`);
        
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => deletePlayer(player.id));
        }
        
        if (recoverBtn) {
            recoverBtn.addEventListener('click', () => markAsRecovered(player.id));
        }
    });
}

// Create HTML for a player card
function createPlayerCard(player) {
    const statusClass = getStatusClass(player);
    const statusIcon = getStatusIcon(player);
    const progressColor = getProgressColor(player.recovery_progress);
    
    return `
        <div class="player-card ${player.is_recovered ? 'recovered' : ''}" data-player-id="${player.id}">
            <div class="player-card-header">
                <div class="player-info">
                    <h3>⚽ ${player.player_name}</h3>
                    <span class="player-meta">${player.age} years • ${player.position}</span>
                </div>
                <div class="player-status ${statusClass}">
                    ${statusIcon} ${player.status}
                </div>
            </div>
            
            <div class="injury-details">
                <div class="injury-badge ${player.injury_severity.toLowerCase()}">
                    ${player.injury_type} • ${player.injury_severity}
                </div>
                <div class="injury-stats">
                    <span>⚠️ ${player.previous_injuries} previous injuries</span>
                    <span>💪 Fitness: ${player.fitness_level}/10</span>
                </div>
            </div>
            
            <div class="recovery-countdown">
                <div class="countdown-main">
                    <div class="countdown-number" id="countdown-${player.id}">
                        ${player.is_recovered ? '✅' : player.days_remaining}
                    </div>
                    <div class="countdown-label">
                        ${player.is_recovered ? 'Recovered' : 'Days Remaining'}
                    </div>
                </div>
                <div class="recovery-dates">
                    <div class="date-item">
                        <span class="date-label">Started:</span>
                        <span class="date-value">${formatDate(player.prediction_date)}</span>
                    </div>
                    <div class="date-item">
                        <span class="date-label">Expected Return:</span>
                        <span class="date-value">${formatDate(player.expected_recovery_date)}</span>
                    </div>
                </div>
            </div>
            
            <div class="recovery-progress-bar">
                <div class="progress-info">
                    <span>Recovery Progress</span>
                    <span class="progress-percent">${player.recovery_progress}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill ${progressColor}" style="width: ${player.recovery_progress}%"></div>
                </div>
            </div>
            
            <div class="prediction-info">
                <div class="info-row">
                    <span>📅 Predicted Recovery:</span>
                    <span class="info-value">${player.predicted_recovery_days} days</span>
                </div>
                <div class="info-row">
                    <span>⚠️ Setback Risk:</span>
                    <span class="risk-badge risk-${player.predicted_setback_risk.toLowerCase()}">
                        ${player.predicted_setback_risk} (${player.setback_probability}%)
                    </span>
                </div>
            </div>
            
            ${player.notes ? `
                <div class="player-notes">
                    <span class="notes-icon">📝</span>
                    <p>${player.notes}</p>
                </div>
            ` : ''}
            
            <div class="player-actions">
                ${!player.is_recovered ? `
                    <button class="btn btn-success btn-sm" id="recover-${player.id}">
                        ✅ Mark as Recovered
                    </button>
                ` : ''}
                <button class="btn btn-danger btn-sm" id="delete-${player.id}">
                    🗑️ Remove
                </button>
            </div>
        </div>
    `;
}

// Update countdowns in real-time
function updateCountdowns() {
    allPlayers.forEach(player => {
        const countdownEl = document.getElementById(`countdown-${player.id}`);
        if (countdownEl && !player.is_recovered) {
            const daysRemaining = calculateDaysRemaining(player.expected_recovery_date);
            countdownEl.textContent = daysRemaining;
            player.days_remaining = daysRemaining;
        }
    });
}

// Calculate days remaining
function calculateDaysRemaining(expectedDate) {
    const today = new Date();
    const expected = new Date(expectedDate);
    
    if (today >= expected) return 0;
    
    const diff = expected - today;
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

// Delete a tracked player
async function deletePlayer(playerId) {
    if (!confirm('Are you sure you want to remove this player from tracking?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/tracking/${playerId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            allPlayers = allPlayers.filter(p => p.id !== playerId);
            filterAndDisplayPlayers();
            showSuccess('Player removed from tracking');
            
            if (allPlayers.length === 0) {
                document.getElementById('emptyState').style.display = 'block';
                document.getElementById('playersGrid').style.display = 'none';
            }
        } else {
            showError(data.error || 'Failed to remove player');
        }
    } catch (error) {
        console.error('Error deleting player:', error);
        showError('An error occurred while removing the player');
    }
}

// Mark player as recovered
async function markAsRecovered(playerId) {
    try {
        const response = await fetch(`/api/tracking/${playerId}/recover`, {
            method: 'PUT'
        });
        
        const data = await response.json();
        
        if (data.success) {
            const player = allPlayers.find(p => p.id === playerId);
            if (player) {
                player.is_recovered = true;
                player.status = 'Recovered';
            }
            filterAndDisplayPlayers();
            showSuccess('Player marked as recovered! 🎉');
        } else {
            showError(data.error || 'Failed to update player status');
        }
    } catch (error) {
        console.error('Error marking player as recovered:', error);
        showError('An error occurred while updating the player');
    }
}

// Helper functions
function getStatusClass(player) {
    if (player.is_recovered) return 'status-recovered';
    if (player.days_remaining === 0) return 'status-complete';
    if (player.days_remaining <= 3) return 'status-almost';
    if (player.recovery_progress >= 75) return 'status-final';
    if (player.recovery_progress >= 50) return 'status-progress';
    return 'status-early';
}

function getStatusIcon(player) {
    if (player.is_recovered) return '✅';
    if (player.days_remaining === 0) return '🎯';
    if (player.days_remaining <= 3) return '⚡';
    if (player.recovery_progress >= 75) return '🔥';
    if (player.recovery_progress >= 50) return '💪';
    return '🏥';
}

function getProgressColor(progress) {
    if (progress >= 90) return 'progress-complete';
    if (progress >= 75) return 'progress-high';
    if (progress >= 50) return 'progress-medium';
    if (progress >= 25) return 'progress-low';
    return 'progress-start';
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
    });
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function showError(message) {
    showNotification(message, 'error');
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

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});
