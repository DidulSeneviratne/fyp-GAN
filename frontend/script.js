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
    
    const selectedColors1 = getSelectedColors(); // Get colors in RGB format
    formData.append("colors1", JSON.stringify(selectedColors1)); // Send as JSON array

    try {
        deleteAllImages();
        // Show the loading indicator
        loadingIndicator.style.display = "block";
        const response = await fetch("http://localhost:8000/api/generate-ui", {
            method: "POST",
            body: formData
        }) 
    
        /* Ensure 'response' exists before checking properties
        if (!response) {
            alert("No response received from server")
            throw new Error("No response received from server");
        }

        if (!response.ok) {
            const errorData = await response.json(); // Read error response
            alert("error")
            throw new Error(`Error ${response.status}: ${errorData.detail || "Unknown error"}`);
        }

        const data = await response.json(); // Read JSON response
        console.log("Success:", data);
        alert(data) */ 

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
            fallbackLoadFromOutputs1();
        }
        
        /* data.generated_images.forEach((imagePath, index) => {
            const tempImage = new Image();
            tempImage.src = `http://localhost:8000/${imagePath}`; // Adjust path to backend URL

            tempImage.onload = function () {
                // Get image dimensions
                const imageWidth = tempImage.naturalWidth;
                const imageHeight = tempImage.naturalHeight;

                // Create a new image element
                const imgElement = document.createElement("img");
                imgElement.src = tempImage.src;
                imgElement.style.width = `${imageWidth / 2}px`;
                imgElement.style.height = `${imageHeight / 2}px`;
                imgElement.style.margin = "10px";

                imageContainer.appendChild(imgElement); // Append image to the container
                loadedImages.push(tempImage.src); // Store successful image path

                console.log(`Loaded: ${tempImage.src} (${imageWidth}x${imageHeight})`);
                
                // Save to localStorage for persistence
                localStorage.setItem("loadedImages", JSON.stringify(loadedImages));

                // Show download button after images load
                const generatedImage = document.getElementById("generatedImage");
                generatedImage.style.display = "none";
                loadingIndicator.style.display = "none";
                downloadBtn.style.display = "inline-block";
            };

            tempImage.onerror = function () {
                console.log(`Failed to load image: ${tempImage.src}`);
            };
        }); */
        
    } catch (error) {
        console.error("Fetch Error:", error);
        alert(error.message);
        /* fallbackLoadFromOutputs();
        fallbackLoadFromOutputs1(); */
    }
});

async function deleteAllImages() {
    try {
        const response = await fetch("http://localhost:8000/delete-all-images", {
            method: "DELETE"
        });

        const result = await response.json();
        console.log(result.message);
        // alert(result.message);  // Show success message

        // Remove all images from the UI
        document.getElementById("imageContainer").innerHTML = "";

    } catch (error) {
        console.error("Error deleting images:", error);
        alert(error);
    }
}

function fallbackLoadFromOutputs1() {

    const imageContainer = document.getElementById("imageContainer");
    let loadedImages = []; 

    for (let i = 0; i < 5; i++) {
        const generatedImagePath = `http://localhost:8001/output/generated_ui${i}.png`;
        const tempImage = new Image();

        tempImage.src = generatedImagePath;

        tempImage.onload = function () {
            const imageWidth = tempImage.naturalWidth;
            const imageHeight = tempImage.naturalHeight;
            const device = document.getElementById("device").value;

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
            } else {
                // Update the <img> element with the correct source and dimensions
                imgElement.style.width = `${imageWidth / 2}px`;
                imgElement.style.height = `${imageHeight / 2}px`;
            }

            imgElement.style.margin = "10px"; // Add some spacing between images

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

    /*downloadBtn.onclick = async function () {
        const zip = new JSZip();  // Create a new zip instance
        const folder = zip.folder("Generated_UI_Images");  // Create a folder in the zip file

        const imageElements = document.querySelectorAll("#imageContainer img");  // Get all displayed images

        if (imageElements.length === 0) {
            alert("No images to download.");
            return;
        }

        // Loop through each image and fetch its data
        for (let i = 0; i < imageElements.length; i++) {
            const img = imageElements[i];
            const imgURL = img.src;
            const fileName = `Generated_UI_${i + 1}.png`;

            try {
                // Fetch the image as a blob
                console.log(`Fetching image: ${imgURL}`);  // Log the image URL
                const response = await fetch(imgURL);

                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }

                const blob = await response.blob();

                // Add the blob to the zip folder
                folder.file(fileName, blob);
                console.log(`Added ${fileName} to ZIP`);
            } catch (error) {
                console.error(`Failed to fetch image ${fileName}:`, error);
            }
        }

        // Generate the ZIP file and trigger the download
        zip.generateAsync({ type: "blob" })
            .then(function (content) {
                const link = document.createElement("a");
                link.href = URL.createObjectURL(content);
                link.download = "Generated_UI_Images.zip";
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            });
    };*/

}

function fallbackLoadFromOutputs() {
    // Assuming the output image is saved with a standard name in the 'outputs' folder
    // const fallbackUrl = "http://localhost:8000/outputs/generated_ui.png";

    /* Display the fallback image
    const generatedImage = document.getElementById("generatedImage");
    generatedImage.src = "../backend/outputs/final_image.jpg";*/

    const generatedImagePath = "../backend/outputs/final_image.jpg"; // Path to the generated image

    // Create a temporary Image object to load and get dimensions
    const tempImage = new Image();

    // Set the source of the temporary image
    tempImage.src = generatedImagePath;

    const generatedImage = document.getElementById("generatedImage");

    // Wait for the image to load
    tempImage.onload = function () {
        // Retrieve the natural width and height of the image
        const imageWidth = tempImage.naturalWidth;
        const imageHeight = tempImage.naturalHeight;
        const device = document.getElementById("device").value;

        if (device == 'Desktop'){
            // Update the <img> element with the correct source and dimensions
            generatedImage.src = generatedImagePath;
            generatedImage.style.width = `${imageWidth/8}px`;
            generatedImage.style.height = `${imageHeight/8}px`;
        } else if (device == 'Tablet') {
            // Update the <img> element with the correct source and dimensions
            generatedImage.src = generatedImagePath;
            generatedImage.style.width = `${imageWidth/4}px`;
            generatedImage.style.height = `${imageHeight/4}px`;
        } else {
            // Update the <img> element with the correct source and dimensions
            generatedImage.src = generatedImagePath;
            generatedImage.style.width = `${imageWidth/2}px`;
            generatedImage.style.height = `${imageHeight/2}px`;
        }

        console.log(`Image dimensions set: ${imageWidth}x${imageHeight}`);
    };

    // Handle errors
    tempImage.onerror = function () {
        alert("Failed to load the generated image.");
        // loadingIndicator.textContent = "Failed to load image. Please try again.";
    };

    localStorage.setItem("generatedImageSrc", generatedImage.src);

    // Show the download button
    const downloadBtn = document.getElementById("downloadBtn");
    downloadBtn.style.display = "inline-block";
    downloadBtn.onclick = function () {
        // Create a temporary download link
        const link = document.createElement("a");
        link.href = generatedImage.src; // Use the image source as the download link
        link.download = "Generated_UI.png"; // Set the download filename
        document.body.appendChild(link); // Add the link to the document
        link.click(); // Trigger the download
        document.body.removeChild(link); // Remove the link after the download
    };

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

/*document.getElementById("downloadBtn").addEventListener("click", function () {
    // Get the image element
    const imageElement = document.getElementById("generatedImage");

    // Ensure the image source (src) is valid
    if (imageElement.src) {
        // Create a temporary <a> element
        const link = document.createElement("a");
        link.href = imageElement.src; // Set the href to the image source
        link.download = "Generated_UI.png"; // Set the download filename

        // Trigger the download
        link.click();
    } else {
        alert("No image found to download!");
    }
});*/

document.getElementById("downloadBtn").onclick = async function () {
    /* const files = document.getElementById("sketch").files;

    if (files.length === 0) {
        alert("Please select some images first.");
        return;
    }

    const zip = new JSZip();
    const folder = zip.folder("Generated_UI_Images");

    // Add each selected file to the zip
    Array.from(files).forEach((file, index) => {
        folder.file(file.name, file);  // Add the file to the zip folder
    });

    // Generate the ZIP and trigger the download
    const content = await zip.generateAsync({ type: "blob" });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(content);
    link.download = "Generated_UI_Images.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link); */

    const zip = new JSZip();
    const folder = zip.folder("Generated_UI_Images");

    // Define the image paths in the outputs folder
    const imagePaths = [
        'http://localhost:8001/output/generated_ui0.png',
        'http://localhost:8001/output/generated_ui1.png',
        'http://localhost:8001/output/generated_ui2.png',
        'http://localhost:8001/output/generated_ui3.png',
        'http://localhost:8001/output/generated_ui4.png'
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

/* document.getElementById("resetForm").addEventListener("click", function () {
    document.getElementById("uploadForm").reset(); // Resets the form fields
    
    // Reset additional UI elements
    document.getElementById("generatedImage").src = "";
    document.getElementById("generatedImage").style.display = "none";
    document.getElementById("downloadBtn").style.display = "none";
    document.getElementById("colorPickerContainer").style.display = "none"; // Hide color picker if shown
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

document.getElementById("submitReview").addEventListener("click", async function () {
    let rating = document.getElementById("rating").value;
    let review = document.getElementById("review").value;

    if (review.trim() === "") {
        alert("Please write a review before submitting.");
        return;
    }

    let newReview = `Rating: ${rating}⭐\nReview: ${review}\nDate: ${new Date().toLocaleString()}\n\n`;

    // alert(newReview);

    // Store or send data (This can be stored in localStorage for now)
    let reviewData = {
        rating: rating,
        review: review,
        date: new Date().toLocaleString(),
    };

    localStorage.setItem("userReview", reviewData);

    try {
        // Request access to the `review.txt` file
        const fileHandle = await window.showOpenFilePicker({
            types: [{ description: "Text Files", accept: { "text/plain": [".txt"] } }],
            excludeAcceptAllOption: true,
            multiple: false
        });

        const file = await fileHandle[0].getFile();
        let text = await file.text(); // Read existing content
        let updatedContent = text + newReview; // Append new review

        // Write the updated content back to the file
        const writable = await fileHandle[0].createWritable();
        await writable.write(updatedContent);
        await writable.close();

        alert("Your review has been saved to review.txt!");

    } catch (error) {
        console.error("Error accessing file:", error);
        alert("Failed to save the review. Please allow file access.");
    }

    // Call function to download the updated text file
    saveToTextFile(reviewData);

    alert("Thank you for your feedback!");
    modal.style.display = "none";
    document.getElementById("review").value = ""; // Clear text area
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
}); */

