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
                
                if(data.result==="FAKE"){
                    alert("DEEPFAKE DETECTED!\nReport on this link: https://eservices.tnpolice.gov.in/CCTNSNICSDC/RedirectToNationalCyberCrimeReportingPortal%28NCRP%29")
                }
                if(data.result==="REAL"){
                    alert("Image is Real.")
                }
                
            })
            .catch(error => console.error('Error:', error));
    }
});
