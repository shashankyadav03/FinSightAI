document.addEventListener('DOMContentLoaded', () => {
    const finetuneForm = document.getElementById('finetune-form');
    const verifyForm = document.getElementById('verify-form');

    if (finetuneForm) {
        finetuneForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const data = document.getElementById('finetune-data').value;
            const response = await fetch('/finetune', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data }),
            });
            const result = await response.json();
            document.getElementById('finetune-result').innerText = JSON.stringify(result, null, 2);
        });
    }

    if (verifyForm) {
        verifyForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const data = document.getElementById('verify-data').value;
            const response = await fetch('/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data }),
            });
            const result = await response.json();
            document.getElementById('verify-result').innerText = JSON.stringify(result, null, 2);
        });
    }
});
