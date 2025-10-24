function play(mediaElement) {
    mediaElement.play();
}

function pause(mediaElement) {
    mediaElement.pause();
}

function stop(mediaElement) {
    mediaElement.pause();
    mediaElement.currentTime = 0;
}

export { play, pause, stop };