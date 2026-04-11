// ======================================================================
// NOVA – Toast Notification Module
// Extracted from script.js for maintainability
// ======================================================================

/**
 * Show a toast notification.
 * @param {string} message - Notification text
 * @param {'info'|'success'|'error'|'warning'} type - Toast type
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
    toast.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ️'}</span><span class="toast-msg">${message}</span>`;
    container.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => toast.classList.add('toast-show'));

    // Remove after 3.5s
    setTimeout(() => {
        toast.classList.remove('toast-show');
        toast.classList.add('toast-hide');
        setTimeout(() => toast.remove(), 350);
    }, 3500);
}
