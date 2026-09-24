/**
 * Placement Prep Knowledge Assistant — Frontend Application Logic
 * Communicates with the FastAPI multi-agent backend.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const statChunks = document.getElementById("stat-chunks");
    const statSeniors = document.getElementById("stat-seniors");
    const statCompanies = document.getElementById("stat-companies");
    const statSources = document.getElementById("stat-sources");
    const companySelect = document.getElementById("company-select");
    const topicPillsContainer = document.getElementById("topic-pills");
    
    const queryInput = document.getElementById("query-input");
    const submitBtn = document.getElementById("submit-btn");
    const loadingState = document.getElementById("loading-state");
    const responseContainer = document.getElementById("response-container");
    
    const conflictBanner = document.getElementById("conflict-banner");
    const answerBody = document.getElementById("answer-body");
    const sourcesPills = document.getElementById("sources-pills");
    const evidenceList = document.getElementById("evidence-list");
    const evidenceCount = document.getElementById("evidence-count");
    const copyBtn = document.getElementById("copy-btn");
    const toast = document.getElementById("toast");

    let selectedTopics = new Set();
    let currentRawAnswer = "";

    // 1. Fetch Stats & Populate Filters on Startup
    async function loadStats() {
        try {
            const resp = await fetch("/api/stats");
            if (!resp.ok) return;
            const data = await resp.json();

            // Populate stats counters
            statChunks.textContent = data.total_chunks || 0;
            statSeniors.textContent = data.total_seniors || 0;
            statCompanies.textContent = data.total_companies || 0;
            statSources.textContent = data.total_sources || 0;

            // Populate company dropdown
            if (data.companies_list && data.companies_list.length > 0) {
                data.companies_list.forEach(comp => {
                    const opt = document.createElement("option");
                    opt.value = comp;
                    opt.textContent = comp;
                    companySelect.appendChild(opt);
                });
            }

            // Populate topic pills
            if (data.topics_list && data.topics_list.length > 0) {
                topicPillsContainer.innerHTML = "";
                data.topics_list.forEach(topic => {
                    const pill = document.createElement("div");
                    pill.className = "topic-pill";
                    pill.textContent = topic;
                    pill.addEventListener("click", () => {
                        if (selectedTopics.has(topic)) {
                            selectedTopics.delete(topic);
                            pill.classList.remove("active");
                        } else {
                            selectedTopics.add(topic);
                            pill.classList.add("active");
                        }
                    });
                    topicPillsContainer.appendChild(pill);
                });
            }
        } catch (err) {
            console.error("Failed to load corpus stats:", err);
        }
    }

    // 2. Execute RAG Query
    async function executeQuery(queryText) {
        if (!queryText || !queryText.trim()) return;

        // UI updates during fetch
        responseContainer.classList.add("hidden");
        loadingState.classList.remove("hidden");
        submitBtn.disabled = true;

        const payload = {
            query: queryText.trim(),
            topic_filters: selectedTopics.size > 0 ? Array.from(selectedTopics) : null,
            company_filter: companySelect.value ? companySelect.value : null
        };

        try {
            const resp = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                throw new Error("Failed to process query.");
            }

            const data = await resp.json();
            renderResponse(data);
        } catch (err) {
            console.error("Query Error:", err);
            answerBody.innerHTML = `<p style="color: #EF4444;">Something went wrong while retrieving and synthesizing. Please try again.</p>`;
            responseContainer.classList.remove("hidden");
        } finally {
            loadingState.classList.add("hidden");
            submitBtn.disabled = false;
        }
    }

    // 3. Render Synthesized Response & Evidence
    function renderResponse(data) {
        currentRawAnswer = data.final_answer || "";

        // Conflict Banner toggle
        if (data.has_conflict) {
            conflictBanner.classList.remove("hidden");
        } else {
            conflictBanner.classList.add("hidden");
        }

        // Render Markdown Answer
        if (window.marked) {
            answerBody.innerHTML = marked.parse(data.final_answer);
        } else {
            answerBody.innerText = data.final_answer;
        }

        // Render Cited Sources Pills
        sourcesPills.innerHTML = "";
        if (data.sources_used && data.sources_used.length > 0) {
            data.sources_used.forEach(src => {
                const pill = document.createElement("span");
                pill.className = "source-pill-item";
                pill.textContent = `📌 ${src}`;
                sourcesPills.appendChild(pill);
            });
        } else {
            sourcesPills.innerHTML = `<span style="color: var(--text-muted); font-size: 0.8rem;">No specific sources recorded.</span>`;
        }

        // Render Retrieved Context Evidence Chunks
        evidenceList.innerHTML = "";
        const chunks = data.retrieved_chunks || [];
        evidenceCount.textContent = chunks.length;

        chunks.forEach((c, idx) => {
            const card = document.createElement("div");
            card.className = "evidence-card";
            
            const senior = c.senior_id || "Senior";
            const source = c.source_name || "File";
            const company = c.company ? ` | Company: ${c.company}` : "";
            const topic = c.topic ? ` | Topic: ${c.topic}` : "";
            const score = c.rerank_score !== undefined ? `Rerank: ${c.rerank_score}` : "";

            card.innerHTML = `
                <div class="evidence-header">
                    <span class="evidence-source">#${idx + 1} — ${senior} (${source})${company}${topic}</span>
                    <span class="evidence-score">${score}</span>
                </div>
                <div class="evidence-text">${escapeHtml(c.text || "")}</div>
            `;
            evidenceList.appendChild(card);
        });

        responseContainer.classList.remove("hidden");
        responseContainer.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    // Helper: Escape HTML in code snippets
    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // 4. Event Listeners
    submitBtn.addEventListener("click", () => {
        executeQuery(queryInput.value);
    });

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            executeQuery(queryInput.value);
        }
    });

    // Suggestion Chips Click
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            queryInput.value = q;
            executeQuery(q);
        });
    });

    // Copy to Clipboard
    copyBtn.addEventListener("click", () => {
        if (!currentRawAnswer) return;
        navigator.clipboard.writeText(currentRawAnswer).then(() => {
            toast.classList.remove("hidden");
            setTimeout(() => {
                toast.classList.add("hidden");
            }, 2500);
        });
    });

    // Initial Load
    loadStats();
});
