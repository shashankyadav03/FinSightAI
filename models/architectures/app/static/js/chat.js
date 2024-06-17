// static/js/chat.js

document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const userMessage = userInput.value.trim();
        if (userMessage) {
            addMessage('You', userMessage);
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
                    addMessage('Model', result.response, result.id);
                } else {
                    addMessage('Error', result.error);
                }
            } catch (error) {
                addMessage('Error', 'An error occurred: ' + error.message);
            }
        }
    });

    function addMessage(sender, message, id) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        if (sender === 'Model' && id) {
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
