// Predefined keywords for diabetes-related topics
const keywords = [
    "diabetes", "type 1 diabetes", "type 2 diabetes", "gestational diabetes", "pre-diabetes",
    "insulin", "glucose", "blood sugar", "hyperglycemia", "hypoglycemia",
    "fatigue", "frequent urination", "thirst", "blurred vision", "weight loss",
    "numbness", "tingling sensation", "neuropathy", "retinopathy", "nephropathy",
    "diabetic ketoacidosis", "cardiovascular disease", "foot ulcers", "skin infections",
    "HbA1c", "A1c", "fasting blood sugar", "glucose tolerance test",
    "continuous glucose monitoring", "blood sugar levels", "self-monitoring of blood glucose",
    "insulin therapy", "oral hypoglycemics", "metformin", "sulfonylureas",
    "GLP-1 receptor agonists", "SGLT2 inhibitors", "DPP-4 inhibitors", "lifestyle changes",
    "diet and exercise", "bariatric surgery", "low glycemic index", "carb counting",
    "diabetic diet", "meal planning", "physical activity", "weight management",
    "stress management", "diabetes education", "glucometer", "insulin pump",
    "diabetes app", "smart insulin pens", "artificial pancreas", "diabetic wearables",
    "low-carb diet", "ketogenic diet", "whole grains", "fiber-rich foods",
    "sugar substitutes", "artificial sweeteners", "healthy fats", "protein-rich foods",
    "diabetes mellitus", "autoimmune disorder", "pancreatic beta cells",
    "insulin resistance", "genetic predisposition", "endocrinology", "diabetes research",
    "diabetes support groups", "diabetes awareness", "peer support", "advocacy",
    "juvenile diabetes", "pediatric diabetes", "pregnancy diabetes",
    "high-risk pregnancy", "maternal glucose"
];

// Function to compute TF-IDF relevance score
function calculateRelevance(text, keywords) {
    // Normalize text: lowercase and remove punctuation
    const normalizedText = text.toLowerCase().replace(/[^\w\s]/g, " ");
    const textWords = normalizedText.split(/\s+/);

    // Count word occurrences
    const wordCount = {};
    textWords.forEach(word => {
        wordCount[word] = (wordCount[word] || 0) + 1;
    });

    // Calculate relevance score based on keyword matches
    let score = 0;
    keywords.forEach(keyword => {
        if (wordCount[keyword]) {
            score += wordCount[keyword];
        }
    });

    // Normalize the score
    return score / textWords.length;
}

// Add event listener to classify button
document.getElementById("classify").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value;

    if (!text.trim()) {
        alert("Please paste some text to classify!");
        return;
    }

    // Debug: Log the entered text
    console.log("Entered Text:", text);

    // Calculate relevance score
    const relevanceScore = calculateRelevance(text, keywords);

    // Debug: Log the relevance score
    console.log("Relevance Score:", relevanceScore);

    if (relevanceScore < 0.1) { // Adjust threshold for relevance
        document.getElementById("result").innerText =
            "The text is irrelevant to diabetes topics.";
        document.getElementById("confidenceChart").style.display = "none"; // Hide chart
        return;
    }

    // If relevant, send to server for classification
    const serverUrl = "http://127.0.0.1:5001/predict";

    try {
        const response = await fetch(serverUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        const result = await response.json();
        console.log("Server response:", result);

        if (result.probabilities) {
            document.getElementById("result").innerText =
                `Prediction: ${result.predicted_label}`;
            renderChart(result.probabilities);
        } else {
            document.getElementById("result").innerText =
                "Error: Unable to classify text.";
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerText =
            "Server error. Please try again later.";
    }
});

// Function to render chart
function renderChart(probabilities) {
    const ctx = document.getElementById("confidenceChart").getContext("2d");

    if (window.confidenceChart) {
        window.confidenceChart.destroy?.();
    }

    // Debug: Log probabilities to confirm valid data
    console.log("Rendering chart with probabilities:", probabilities);

    // Validate probabilities
    if (typeof probabilities.real === "undefined" || typeof probabilities.false === "undefined") {
        console.error("Invalid probabilities:", probabilities);
        document.getElementById("confidenceChart").style.display = "none"; // Hide chart
        alert("Error: Unable to display chart due to missing data.");
        return;
    }

    window.confidenceChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Real", "False"],
            datasets: [{
                label: "Confidence Scores",
                data: [probabilities.real, probabilities.false],
                backgroundColor: ["#4caf50", "#f44336"], // Green for Real, Red for False
                borderWidth: 1,
            }],
        },
        options: {
            plugins: {
                legend: {
                    labels: {
                        generateLabels: (chart) => {
                            // Customize legend to show label without color box
                            return chart.data.datasets.map((dataset) => ({
                                text: dataset.label, // Only the label text is displayed
                                fillStyle: "transparent", // Make the color box transparent
                                strokeStyle: "transparent", // No border around the box
                            }));
                        },
                    },
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100, // Assuming percentages for confidence scores
                },
            },
        },
    });
    
    // Show the chart
    document.getElementById("confidenceChart").style.display = "block";
    
}


