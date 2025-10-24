// This file initializes the media player application, sets up event listeners, and manages the overall application state.

import { play, pause, stop } from './player.js';
import { addToPlaylist, getPlaylist } from './playlist.js';
import { updateControls } from './controls.js';

document.addEventListener('DOMContentLoaded', () => {
    const playButton = document.getElementById('playButton');
    const pauseButton = document.getElementById('pauseButton');
    const stopButton = document.getElementById('stopButton');
    const playlistContainer = document.getElementById('playlistContainer');

    playButton.addEventListener('click', () => {
        play();
        updateControls();
    });

    pauseButton.addEventListener('click', () => {
        pause();
        updateControls();
    });

    stopButton.addEventListener('click', () => {
        stop();
        updateControls();
    });

    const playlist = getPlaylist();
    playlist.forEach(item => {
        const listItem = document.createElement('li');
        listItem.textContent = item.title;
        playlistContainer.appendChild(listItem);
    });
});