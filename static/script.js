document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    function addMessage(content, type = 'system') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        if (type === 'user') {
            messageDiv.textContent = content;
        } else if (type === 'response') {
            // Add preferences in a compact format
            const preferencesDiv = document.createElement('div');
            preferencesDiv.className = 'preferences';
            preferencesDiv.innerHTML = content.preferences;
            messageDiv.appendChild(preferencesDiv);

            // Add recommendations in a compact format
            content.recommendations.forEach(city => {
                const cityDiv = document.createElement('div');
                cityDiv.className = 'city-recommendation';
                cityDiv.innerHTML = `
                    <div class="city-header">
                        <h3>🏆 ${city.name}</h3>
                    </div>
                    <div class="city-content">
                        ${city.details.map(detail => `<p>${detail}</p>`).join('')}
                    </div>
                `;
                messageDiv.appendChild(cityDiv);
            });
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function addLoadingMessage() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message loading';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return loadingDiv;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Disable input and button while processing
        userInput.disabled = true;
        sendButton.disabled = true;

        // Add user message to chat
        addMessage(message, 'user');

        // Add loading indicator
        const loadingMessage = addLoadingMessage();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            // Remove loading indicator
            loadingMessage.remove();

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            addMessage(data, 'response');
        } catch (error) {
            loadingMessage.remove();
            addMessage('Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.', 'error');
            console.error('Error:', error);
        } finally {
            // Clear and re-enable input
            userInput.value = '';
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});
 have