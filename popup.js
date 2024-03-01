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
                // Check if the response is successful
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                // Handle different content types
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    return response.json();
                } else {
                    return response.text();
                }
            })
            .then(data => {
                // Update the result div with the received data
                if(data.result==="FAKE"){
                    alert("FAKE DETECTED!")
                }
                document.getElementById('result').innerText = `Result: ${data.result}`;
            })
            .catch(error => console.error('Error:', error));
    }
});
