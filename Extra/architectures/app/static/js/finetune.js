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
