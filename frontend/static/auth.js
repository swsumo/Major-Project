/**
 * FitAI Auth Helper
 * Add this script to every HTML page:
 * <script src="static/auth.js"></script>
 *
 * Replaces direct fetch() calls with authenticated calls.
 * Usage: authFetch('/api/user/1/plan') instead of fetch(...)
 */

const API_URL =
    window.location.hostname.includes("localhost")
        ? "http://localhost:5000/api"
        : "https://major-project-20zc.onrender.com/api";
function getToken() {
    return localStorage.getItem('token');
}

function getUserId() {
    return localStorage.getItem('user_id');
}

function isLoggedIn() {
    return !!getToken() && !!getUserId();
}

function logout() {
    localStorage.clear();
    window.location.href = 'login.html';
}

/**
 * Authenticated fetch — automatically adds JWT Bearer token.
 * Use this instead of fetch() for all API calls.
 */
async function authFetch(url, options = {}) {
    const token = getToken();

    const headers = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(`${API_URL}${url}`, {
        ...options,
        headers
    });

    // If token expired, redirect to login
    if (response.status === 401) {
        const data = await response.json();
        if (data.message && data.message.includes('expired')) {
            alert('Session expired. Please login again.');
            logout();
            return null;
        }
    }

    return response;
}

// Guard — redirect to login if not authenticated
function requireAuth() {
    if (!isLoggedIn()) {
        window.location.href = 'login.html';
    }
}