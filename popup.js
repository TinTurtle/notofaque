document.addEventListener("DOMContentLoaded",()=>{
    const toggleButton = document.querySelector("toggleButton")
    const snipButton = document.querySelector("snipButton")

    toggleButton.addEventListener('click', () => {
        chrome.runtime.sendMessage({ action: 'scan' });
    });
    snipButton.addEventListener('click',()=>{
        chrome.runtime.sendMessage({ action: 'snip' });
    });
})