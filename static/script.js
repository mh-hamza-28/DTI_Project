document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const smiles = document.getElementById('smiles').value.trim();
    const sequence = document.getElementById('sequence').value.trim();
    const btn = document.getElementById('submit-btn');
    const resultBox = document.getElementById('result-container');
    const errorBox = document.getElementById('error-container');

    // Reset state
    resultBox.classList.add('hidden');
    errorBox.classList.add('hidden');
    btn.disabled = true;
    btn.textContent = 'Analyzing Features & Predicting...';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles, sequence })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to predict interaction. Ensure valid sequences are provided.');
        }

        // Populate results
        const labelEl = document.getElementById('prediction-label');
        labelEl.textContent = data.interaction;
        labelEl.style.color = data.interaction === 'Interacts' ? 'var(--success)' : 'var(--error)';

        document.getElementById('confidence-score').textContent = data.confidence.toFixed(1) + '%';
        document.getElementById('lr-score').textContent = (data.lr_score * 100).toFixed(1) + '%';
        document.getElementById('rf-score').textContent = (data.rf_score * 100).toFixed(1) + '%';

        resultBox.classList.remove('hidden');
    } catch (err) {
        errorBox.textContent = err.message;
        errorBox.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Interaction';
    }
});
