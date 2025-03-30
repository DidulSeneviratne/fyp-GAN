const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 8001;

const server = http.createServer((req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    // Handle preflight OPTIONS request
    if (req.method === "OPTIONS") {
        res.writeHead(204);
        res.end();
        return;
    }

    if (req.method === "POST" && req.url === "/submit-review") {
        let body = "";

        req.on("data", chunk => {
            body += chunk.toString();
        });

        req.on("end", () => {
            const { rating, review } = JSON.parse(body);
            const entry = `Rating: ${rating}\nReview: ${review}\n---\n`;

            const filePath = path.join(__dirname, "review.txt");

            fs.appendFile(filePath, entry, err => {
                if (err) {
                    res.writeHead(500, { "Content-Type": "text/plain" });
                    res.end("Failed to save review.");
                    return;
                }

                res.writeHead(200, { "Content-Type": "text/plain" });
                res.end("Review submitted successfully!");
            });
        });
    } else {
        res.writeHead(404, { "Content-Type": "text/plain" });
        res.end("Not Found");
    }
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
