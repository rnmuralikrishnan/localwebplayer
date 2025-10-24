// This file manages the playlist functionality, including adding, removing, and navigating through media items.

let playlist = [];

function addToPlaylist(mediaItem) {
    playlist.push(mediaItem);
}

function removeFromPlaylist(index) {
    if (index > -1 && index < playlist.length) {
        playlist.splice(index, 1);
    }
}

function getPlaylist() {
    return playlist;
}

function clearPlaylist() {
    playlist = [];
}

function getCurrentMedia() {
    return playlist.length > 0 ? playlist[0] : null; // Assuming the first item is the current one
}

export { addToPlaylist, removeFromPlaylist, getPlaylist, clearPlaylist, getCurrentMedia };