
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab)=>{
    if(changeInfo.status === "complete" && /^http/.test(tab.url)){
        chrome.scripting.executeScript({
            target: {tabId},
            files: ["./contentScript.js"]
        }).then(()=>{
            console.log("we have injected the content script")
        }).catch(err=> console.log(err, "error in background script line 10"))
    }
})








const FUNC = {
    ALREADY_INJECTED: 'alreadyInjected',
    EXTRACT_SELECTED_IMAGE: 'extractSelectedImage',
    LOG: 'log',
    ALERT: 'alert'
}
const PERSON = {
    REAL: 'Real Photo',
    FAKE: 'Fake Photo'
};

chrome.runtime.onInstalled.addListener(() => {
	chrome.contextMenus.create({
		id: 'deepper',
		title: 'Check for DeepFake', 
		contexts: ['image','video','audio']
	});
});

chrome.contextMenus.onClicked.addListener(( info, tab ) => {
    if (info.menuItemId == 'DeepFakeChecker') {
      // checks, if content.js is already injected, otherwise content.js is added
      chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { func: FUNC.ALREADY_INJECTED }, (response) => {
          if (response === undefined) {
            chrome.tabs.executeScript({ file: 'content.js' });
            console.log('Script added');
          } else {
            console.log('Script was already injected');
            chrome.tabs.captureVisibleTab(null, { format: 'jpeg' }, (data) => {
              chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                chrome.tabs.sendMessage(tab.id, { func: FUNC.EXTRACT_SELECTED_IMAGE, data }, ({ base64, image, error }) => {
                  
                });
              });
           });
          }
        });
      });
    }
});