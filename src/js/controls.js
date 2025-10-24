function updatePlayButton(isPlaying) {
    const playButton = document.getElementById('play-button');
    playButton.textContent = isPlaying ? 'Pause' : 'Play';
}

function updateSkipButton(isAtEnd) {
    const skipButton = document.getElementById('skip-button');
    skipButton.disabled = isAtEnd;
}

function setupControls() {
    const playButton = document.getElementById('play-button');
    const skipButton = document.getElementById('skip-button');

    playButton.addEventListener('click', () => {
        const isPlaying = playButton.textContent === 'Pause';
        updatePlayButton(!isPlaying);
        // Call play/pause function from player.js
    });

    skipButton.addEventListener('click', () => {
        // Call skip function from player.js
    });
}

export { updatePlayButton, updateSkipButton, setupControls };