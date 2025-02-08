document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const sketch = document.getElementById("sketch").files;
    const region = document.getElementById("region").value;
    const age = document.getElementById("age").value;
    const device = document.getElementById("device").value;
    const product = document.getElementById("product").value;

    const loadingIndicator = document.getElementById("loadingIndicator");

    if (sketch.length === 0) {
        alert("Please upload at least one sketch image.");
        return;
    }

    if (sketch.length > 5) {
        alert("You can upload a maximum of 5 images at a time.");
        return;
    }

    const formData = new FormData();
    Array.from(sketch).forEach(file => {
        formData.append("sketch", file);  // Append each file correctly
    });
    //formData.append("sketch", sketch);
    formData.append("region", region);
    formData.append("age", age);
    formData.append("device", device);
    formData.append("product", product);

    try {
        deleteAllImages();
        // Show the loading indicator
        loadingIndicator.style.display = "block";
        const response = await fetch("http://localhost:8000/api/generate-ui", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            /*const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            // Display the generated image
            const generatedImage = document.getElementById("generatedImage");
            generatedImage.src = imageUrl;

            // Show the download button
            const downloadBtn = document.getElementById("downloadBtn");
            downloadBtn.style.display = "inline-block";
            downloadBtn.onclick = function () {
                const link = document.createElement("a");
                link.href = imageUrl;
                link.download = "Generated_UI.png";
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            };*/
            const result = await response.json(); // Expect a JSON response
            console.log("Generated results:", result);

            // Display the generated images
            const resultContainer = document.getElementById("resultContainer");
            resultContainer.innerHTML = ""; // Clear previous results

            result.generated_images.forEach((imageUrl, index) => {
                const imgElement = document.createElement("img");
                imgElement.src = imageUrl;
                imgElement.alt = `Generated UI ${index + 1}`;
                imgElement.style.margin = "10px";
                imgElement.style.maxWidth = "200px";
                resultContainer.appendChild(imgElement);
            });
        } else {
            alert("Failed to generate UI. Please try again.");
        }
    } catch (error) {
        alert(error);
        console.error("Error:", error);
        /* const generatedImage = document.getElementById("generatedImage");
        generatedImage.style.display = "none";
        loadingIndicator.style.display = "none";
        //fallbackLoadFromOutputs();
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
        alert(result.message);  // Show success message

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
        const generatedImagePath = `http://localhost:8001/outputs/generated_ui${i}.png`;
        const tempImage = new Image();

        tempImage.src = generatedImagePath;

        tempImage.onload = function () {
            const imageWidth = tempImage.naturalWidth;
            const imageHeight = tempImage.naturalHeight;

            // Create a new image element for each successful load
            const imgElement = document.createElement("img");
            imgElement.src = generatedImagePath;
            imgElement.style.width = `${imageWidth / 2}px`;
            imgElement.style.height = `${imageHeight / 2}px`;
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

        // Update the <img> element with the correct source and dimensions
        generatedImage.src = generatedImagePath;
        generatedImage.style.width = `${imageWidth/2}px`;
        generatedImage.style.height = `${imageHeight/2}px`;

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
        'http://localhost:8000/outputs/generated_ui0.png',
        'http://localhost:8000/outputs/generated_ui1.png',
        'http://localhost:8000/outputs/generated_ui2.png',
        'http://localhost:8000/outputs/generated_ui3.png',
        'http://localhost:8000/outputs/generated_ui4.png'
    ];

    let imagesAdded = 0;

    for (let i = 0; i < imagePaths.length; i++) {
        const imgURL = imagePaths[i];
        const fileName = `Generated_UI_${i + 1}.png`;

        try {
            const response = await fetch(imgURL);
            
            if (!response.ok) {
                alert('Image not found or failed to load');
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
