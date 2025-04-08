
function togglePasswordVisibility(inputId, toggleElement) {
    var passwordInput = document.getElementById(inputId);
    var toggleButton = toggleElement.querySelector('i');

    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        toggleButton.classList.remove('fa-eye-slash');
        toggleButton.classList.add('fa-eye');
    } else {
        passwordInput.type = 'password';
        toggleButton.classList.remove('fa-eye');
        toggleButton.classList.add('fa-eye-slash');
    }
}


function checkServerStatus() {
    document.getElementById('server-indicator_text').textContent = 'Connecting...';
    document.getElementById('server-indicator').classList.remove("connected");
    document.getElementById('refresh-button').classList.add("connecting");

    fetch('http://localhost:8000/check-status')
        .then(response => {
            if (response.ok) {
                document.getElementById('server-indicator_text').textContent = 'Connected';
                document.getElementById('server-indicator').classList.add("connected");
                document.getElementById('refresh-button').classList.remove("connecting");
                loginInfo.username = getUsername()
                saveLoginInfo()
                sendUpdateName(loginInfo)
            } else {
                // Proxy is OFF
                document.getElementById('server-indicator_text').textContent = "Server is down. Try again later";
                document.getElementById('refresh-button').classList.remove("connecting");

            }
        })
        .catch(error => {
            document.getElementById('server-indicator_text').textContent = 'Proxy is off. Launch it first.';
            document.getElementById('refresh-button').classList.remove("connecting");
            console.error('Error while checking server status:', error);
        });
}
