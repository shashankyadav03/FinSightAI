// static/js/script.js

// static/js/script.js

document.addEventListener('DOMContentLoaded', () => {
    const finetuneForm = document.getElementById('finetune-form');
    const messageDiv = document.getElementById('message');

    finetuneForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        messageDiv.innerHTML = '';

        const formData = new FormData(finetuneForm);
        try {
            const response = await fetch('/finetune', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (response.ok) {
                messageDiv.innerHTML = `<p class="success">${result.message}</p>`;
            } else {
                messageDiv.innerHTML = `<p class="error">${result.error}</p>`;
            }
        } catch (error) {
            messageDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
        }
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const verifyForm = document.getElementById('verify-form');
    const resultsDiv = document.getElementById('results');

    verifyForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        resultsDiv.innerHTML = '';

        const formData = new FormData(verifyForm);
        const input = formData.get('input');
        try {
            const response = await fetch('/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ input })
            });
            const result = await response.json();
            if (response.ok) {
                resultsDiv.innerHTML = `<pre class="success">${JSON.stringify(result.results, null, 2)}</pre>`;
            } else {
                resultsDiv.innerHTML = `<p class="error">${result.error}</p>`;
            }
        } catch (error) {
            resultsDiv.innerHTML = `<p class="error">An error occurred: ${error.message}</p>`;
        }
    });
});

