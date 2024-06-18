// static/js/chat.js

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage('You', userMessage, 'user-message');
            userInput.value = '';
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: userMessage })
                });
                const result = await response.json();
                if (response.ok) {
                    addMessage('Model', result.response, 'model-message');
                } else {
                    addMessage('Error', result.error, 'error-message');
                }
            } catch (error) {
                addMessage('Error', 'An error occurred: ' + error.message, 'error-message');
            }
        }
    });

    function addMessage(sender, message, className) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', className);
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        if (sender === 'Model') {
            const verifyButton = document.createElement('button');
            verifyButton.innerText = 'Verify';
            verifyButton.onclick = () => {
                window.location.href = `/verify?output=${encodeURIComponent(message)}`;
            };
            messageElement.appendChild(verifyButton);
        }
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
