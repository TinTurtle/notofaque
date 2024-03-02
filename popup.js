// popup.js
document.querySelector('form').addEventListener('submit', (event) => {
    event.preventDefault();

    const fileInput = document.querySelector('input[name="file"]');
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    return response.json();
                } else {
                    return response.text();
                }
            })
            .then(data => {
                if (data.result === "FAKE") {
                    document.getElementById('reportButton').style.display = 'block';
                    document.getElementById('reportButton').addEventListener('click', () => {
                        // Open a new tab with the specified report link
                        window.open('https://eservices.tnpolice.gov.in/CCTNSNICSDC/RedirectToNationalCyberCrimeReportingPortal%28NCRP%29', '_blank');
                    });
                    alert("DEEPFAKE DETECTED!\nA report button is available for reporting.");
                }
                if (data.result === "REAL") {
                    alert("Image is Real.");
                }
            })
            .catch(error => console.error('Error:', error));
    }
});
