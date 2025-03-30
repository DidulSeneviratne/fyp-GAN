let hasDeletedImages = false;

document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const sketch = document.getElementById("sketch").files;
    const region = document.getElementById("region").value;
    const age = document.getElementById("age").value;
    const device = document.getElementById("device").value;
    const product = document.getElementById("product").value;
    const useCustomColor = document.getElementById("useCustomColor").checked;

    const loadingIndicator = document.getElementById("loadingIndicator");

    if (sketch.length === 0) {
        alert("Please upload at least one sketch image.");
        return;
    }

    if (sketch.length > 5) {
        alert("You can upload a maximum of 5 images at a time.");
        return;
    }

    function hexToRgb(hex) {
        hex = hex.replace(/^#/, ''); // Remove #
        let r = parseInt(hex.substring(0, 2), 16);
        let g = parseInt(hex.substring(2, 4), 16);
        let b = parseInt(hex.substring(4, 6), 16);
        return `${r}, ${g}, ${b}`;
    }

    function getSelectedColors() {
        return Array.from(document.querySelectorAll("#color-picker1"))
            .map(input => hexToRgb(input.value)); // Convert Hex to RGB
    }

    const formData = new FormData();
    const imageContainer = document.getElementById("imageContainer");
    const downloadBtn = document.getElementById("downloadBtn");
    const orientationRadios = document.querySelectorAll('input[name="orientation"]');
    const selectedOrientation = Array.from(orientationRadios).find(r => r.checked);

    imageContainer.innerHTML = ""; // Clear previous images
    downloadBtn.style.display = "none"; // Hide download button initially

    Array.from(sketch).forEach(file => {
        formData.append("sketch", file);  // Append each file correctly
    });
    //formData.append("sketch", sketch);
    formData.append("region", region);
    formData.append("age", age);
    formData.append("device", device);
    formData.append("product", product);
    formData.append("useCustomColor", useCustomColor);
    if (selectedOrientation) {
        formData.append("orientation", selectedOrientation.value);
    } else {
        formData.append("orientation", "None"); // Send "None" for Desktop or unset
    }
    
    const selectedColors1 = getSelectedColors(); // Get colors in RGB format
    formData.append("colors1", JSON.stringify(selectedColors1)); // Send as JSON array

    try {
        hasDeletedImages = false;
        await deleteAllImages();
        // Show the loading indicator
        loadingIndicator.style.display = "block";
        const response = await fetch("http://localhost:8000/api/generate-ui", {
            method: "POST",
            body: formData
        }) 

        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${await response.text()}`);
        }

        const data = await response.json();
        console.log("Received response:", data);

        if (!data.generated_images || data.generated_images.length === 0) {
            throw new Error("No images were generated.");
        }

        let loadedImages = []; // Store successfully loaded images

        if (response.ok) {
            const generatedImage = document.getElementById("generatedImage");
            generatedImage.style.display = "none";
            loadingIndicator.style.display = "none";
            fallbackLoadFromOutputs();
        }
        
    } catch (error) {
        console.error("Fetch Error:", error);
        alert(error.message);
    }
});

async function deleteAllImages() {
    if (hasDeletedImages) return;

    try {
        const response = await fetch("http://localhost:8000/delete-all-images", {
            method: "DELETE"
        });

        const result = await response.json();
        console.log(result.message);
        // alert(result.message);  // Show success message

        // Remove all images from the UI
        document.getElementById("imageContainer").innerHTML = "";

        hasDeletedImages = true;
    } catch (error) {
        console.error("Error deleting images:", error);
        alert(error);
    }
}

function fallbackLoadFromOutputs() {

    const imageContainer = document.getElementById("imageContainer");
    let loadedImages = []; 

    for (let i = 0; i < 5; i++) {
        const generatedImagePath = `http://localhost:8003/output/generated_ui${i}.png`;
        const tempImage = new Image();

        tempImage.src = generatedImagePath;

        tempImage.onload = function () {
            const imageWidth = tempImage.naturalWidth;
            const imageHeight = tempImage.naturalHeight;
            const device = document.getElementById("device").value;
            const orientationRadios = document.querySelectorAll('input[name="orientation"]');
            const selectedOrientation = Array.from(orientationRadios).find(r => r.checked);

            // Create a new image element for each successful load
            const imgElement = document.createElement("img");
            imgElement.src = generatedImagePath;

            if (device == 'Desktop'){
                // Update the <img> element with the correct source and dimensions
                imgElement.style.width = `${imageWidth / 4}px`;
                imgElement.style.height = `${imageHeight / 4}px`;
            } else if (device == 'Tablet') {
                // Update the <img> element with the correct source and dimensions
                imgElement.style.width = `${imageWidth / 4}px`;
                imgElement.style.height = `${imageHeight / 4}px`;
            } else if (device == 'Mobile' && selectedOrientation.value == 'Landscape'){
                // Update the <img> element with the correct source and dimensions
                imgElement.style.width = `${imageWidth / 4}px`;
                imgElement.style.height = `${imageHeight / 4}px`;
            } else {
                // Update the <img> element with the correct source and dimensions
                imgElement.style.width = `${imageWidth / 2}px`;
                imgElement.style.height = `${imageHeight / 2}px`;
            }

            imgElement.style.margin = "10px"; // Add some spacing between images

            imageContainer.style.display = "inline-block";

            imageContainer.appendChild(imgElement);

            // Save the successful image path to the array
            loadedImages.push(generatedImagePath);

            console.log(`Loaded: ${generatedImagePath} with dimensions ${imageWidth}x${imageHeight}`);
        
            // Save to localStorage after each successful load
            localStorage.setItem("loadedImages", JSON.stringify(loadedImages));
        };

        tempImage.onerror = function () {
            console.log(`Image not found: ${generatedImagePath}`);
            // You can optionally show a placeholder or skip
        };
    }

    const downloadBtn = document.getElementById("downloadBtn");
    downloadBtn.style.display = "inline-block";

}

// Add a new state to the browser's history
// history.pushState({ page: 1 }, "Title", "?page=1");

// Listen for the back button event
window.addEventListener("popstate", function (event) {
    if (event.state) {
        console.log("Navigated back to:", event.state);
        fallbackLoadFromOutputs();
    }
});

document.getElementById('sketch').addEventListener('change', function(event) {
    if (this.files.length > 5) {
        alert('You can only upload a maximum of 5 files.');
        this.value = ''; // Clear the selected files
    }
});


document.getElementById("downloadBtn").onclick = async function () {

    const zip = new JSZip();
    const folder = zip.folder("Generated_UI_Images");

    // Define the image paths in the outputs folder
    const imagePaths = [
        'http://localhost:8003/output/generated_ui0.png',
        'http://localhost:8003/output/generated_ui1.png',
        'http://localhost:8003/output/generated_ui2.png',
        'http://localhost:8003/output/generated_ui3.png',
        'http://localhost:8003/output/generated_ui4.png'
    ];

    let imagesAdded = 0;

    for (let i = 0; i < imagePaths.length; i++) {
        const imgURL = imagePaths[i];
        const fileName = `Generated_UI_${i + 1}.png`;

        try {
            const response = await fetch(imgURL);
            
            if (!response.ok) {
                // alert('Image not found or failed to load');
                console.warn(`Image not found or failed to load: ${imgURL}`);
                continue;  // Skip if the image doesn't exist
            }

            const blob = await response.blob();
            folder.file(fileName, blob);  // Add the image to the zip
            imagesAdded++;
        } catch (error) {
            alert(error);
            console.error(`Failed to fetch ${fileName}:`, error);
        }
    }

    if (imagesAdded === 0) {
        alert("No images found to download.");
        return;
    }

    // Generate and download the ZIP file
    zip.generateAsync({ type: "blob" })
        .then(function (content) {
            const link = document.createElement("a");
            link.href = URL.createObjectURL(content);
            link.download = "Generated_UI_Images.zip";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
};

document.addEventListener("DOMContentLoaded", function () {
    const useCustomColor = document.getElementById("useCustomColor");

    // Toggle color picker visibility when checkbox is clicked
    useCustomColor.addEventListener("change", function () {
        colorPickerContainer.style.display = useCustomColor.checked ? "block" : "none";
    });

});


document.getElementById("device").addEventListener("change", function () {
    const sketches = document.getElementById("sketch").files;
    const device = this.value;

    let imageValidationPromises = [];

    for (let i = 0; i < sketches.length; i++) {
        const sketch = sketches[i];
        const img = new Image();
        img.src = URL.createObjectURL(sketch);

        let promise = new Promise((resolve, reject) => {
            img.onload = function () {
                const width = img.width;
                const height = img.height;
                let validSize = false;

                if (device === "Desktop") {
                    validSize = width >= 1024 && width <= 5120 && height >= 768 && height <= 2880;
                } else if (device === "Mobile") {
                    validSize = width >= 320 && width <= 1440 && height >= 480 && height <= 3200;
                } else if (device === "Tablet") {
                    validSize = width >= 768 && width <= 2560 && height >= 1024 && height <= 1600;
                }

                URL.revokeObjectURL(img.src); // Clean up object URL

                if (!validSize) {
                    alert("The images you are entering are not valid for the device size. Please see the instructions, otherwise the system will automatically resize the image.")
                    resolve(false);
                } else {
                    resolve(true);
                }
            };
        });

        if (!isValid) {
            break; // Stop loop if an invalid image is found
        }

    }

});

document.getElementById("resetForm").addEventListener("click", function () {
    document.getElementById("uploadForm").reset(); // Resets the form fields
    
    // Reset additional UI elements
    document.getElementById("generatedImage").style.display = "inline-block";
    document.getElementById("imageContainer").style.display = "none";
    document.getElementById("downloadBtn").style.display = "none";
    document.getElementById("colorPickerContainer").style.display = "none"; // Hide color picker if shown
    document.getElementById("orientationOptions").style.display = "block"; // Hide color picker if shown
});

let modal = document.getElementById("reviewModal");
let btn = document.getElementById("rateReviewBtn");
let span = document.getElementsByClassName("close")[0];

// Open modal when button is clicked
btn.onclick = function () {
    modal.style.display = "block";
};

// Close modal when (×) is clicked
span.onclick = function () {
    modal.style.display = "none";
};

// Close modal when clicking outside of the content box
window.onclick = function (event) {
    if (event.target === modal) {
        modal.style.display = "none";
    }
};


document.getElementById("submitReview").addEventListener("click", () => {
    const rating = document.getElementById("rating").value;
    const review = document.getElementById("review").value;

    if (!review.trim()) {
        alert("Please enter a review.");
        return;
    }

    fetch("http://localhost:8001/submit-review", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ rating, review }),
    })
    .then(response => response.text())
    .then(data => {
        alert(data); // show confirmation
        document.getElementById("review").value = "";
        document.getElementById("rating").value = "5";
    })
    .catch(error => {
        console.error("Error submitting review:", error);
    });
});

// Function to save and update the text file
function saveToTextFile(data) {
    let blob = new Blob([data], { type: "text/plain" });
    let link = document.createElement("a");

    link.href = URL.createObjectURL(blob);
    link.download = "user_reviews.txt"; // File name
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

let menuButton = document.getElementById("menu");
let dropdownMenu = document.getElementById("dropdownMenu");

let aboutUsModal = document.getElementById("aboutUsModal");
let instructionModal = document.getElementById("instructionModal");

let aboutUsBtn = document.getElementById("aboutUsBtn");
let instructionBtn = document.getElementById("instructionBtn");

let closeButtons = document.querySelectorAll(".close");

// Toggle dropdown menu
menuButton.addEventListener("click", function () {
    let rect = menuButton.getBoundingClientRect(); // Get button position

    let leftOffset = 20; // Adjust left positioning
    let topOffset = 20;  // Adjust top positioning

    dropdownMenu.style.left = `${rect.left - leftOffset}px`; // Set menu position
    dropdownMenu.style.top = `${rect.bottom + window.scrollY - topOffset}px`; // Position below button
    dropdownMenu.style.display = "block";
});

// Open About Us Modal
aboutUsBtn.addEventListener("click", function () {
    aboutUsModal.style.display = "flex";
    dropdownMenu.style.display = "none"; // Hide menu
});

// Open Instruction Modal
instructionBtn.addEventListener("click", function () {
    instructionModal.style.display = "flex";
    dropdownMenu.style.display = "none"; // Hide menu
});

// Close modals when clicking the close button
closeButtons.forEach(btn => {
    btn.addEventListener("click", function () {
        aboutUsModal.style.display = "none";
        instructionModal.style.display = "none";
    });
});

// Close modals when clicking outside the content
window.addEventListener("click", function (event) {
    if (event.target === aboutUsModal) {
        aboutUsModal.style.display = "none";
    }
    if (event.target === instructionModal) {
        instructionModal.style.display = "none";
    }
});

document.addEventListener("click", function (event) {
    if (!menuButton.contains(event.target) && !dropdownMenu.contains(event.target)) {
        dropdownMenu.style.display = "none";
    }
});

document.getElementById("device").addEventListener("change", function () {
    const device = this.value;
    const orientationContainer = document.getElementById("orientationOptions");

    if (device === "Mobile") {
        orientationContainer.style.display = "block";
        document.getElementById("portrait").checked = true;
    } else if (device === "Tablet") {
        orientationContainer.style.display = "block";
        document.getElementById("landscape").checked = true;
    } else {
        orientationContainer.style.display = "none";
        document.querySelectorAll('input[name="orientation"]').forEach(el => el.checked = false);
    }
});