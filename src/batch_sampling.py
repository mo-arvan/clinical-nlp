import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

export_path = "artifacts/results/batch_four_validation.json"

# Define feature extraction patterns
FEATURE_PATTERNS = {
    # Specimen Quality
    "specimen_quality": [
        "satisfactory for evaluation",
        "unsatisfactory for evaluation",
        "transformation zone component not identified",
    ],
    # Normal/Benign Results
    "normal": [
        "negative for intraepithelial lesion",
        "nilm",
        "negative/normal",
        "negative for carcinoma",
        "benign",
    ],
    # Low Grade Changes
    "low_grade": [
        "atypical squamous cells of undetermined significance",
        "asc-us",
        "ascus",
        "low-grade squamous intraepithelial lesion",
        "lsil",
        "cin 1",
    ],
    # High Grade Changes
    "high_grade": [
        "high-grade squamous intraepithelial lesion",
        "hsil",
        "hgsil",
        "atypical squamous cells cannot exclude hsil",
        "asc-h",
        "cin 2",
        "cin 3",
        "cin 2-3",
    ],
    # Glandular Abnormalities
    "glandular": [
        "atypical glandular cells",
        "atypical endocervical cells",
        "atypical endometrial cells",
        "endocervical adenocarcinoma in situ",
        "atypical endocervical cells favors neoplastic",
        "atypical glandular cells favors neoplastic",
        "ais",
    ],
    # Cancer
    "cancer": [
        "squamous cell carcinoma",
        "adenocarcinoma",
        "small cell cancer",
        "endocervical adenocarcinoma",
        "endometrial adenocarcinoma",
        "extrauterine adenocarcinoma",
        "adenocarcinoma nos",
        "endometrial cancer",
        "cancer nos",
    ],
    # Organisms/Infections
    "organisms": [
        "trichomonas vaginalis",
        "candida",
        "bacterial vaginosis",
        "actinomyces",
        "herpes simplex virus",
        "cytomegalovirus",
    ],
    # HPV Status (simplified)
    "hpv_status": [
        "hpv positive",
        "hpv negative",
        "hpv 16 positive",
        "hpv 16 negative",
        "hpv 18 positive",
        "hpv 18 negative",
        "hpv 18/45 positive",
        "hpv 18/45 negative",
        "hpv other positive",
        "hpv other negative",
    ],
}


def extract_features(text):
    """Extract binary features from report text"""
    text = text.lower()
    features = []
    for patterns in FEATURE_PATTERNS.values():
        features.append(1 if any(p in text for p in patterns) else 0)
    return features


with open(export_path, "r") as f:
    results = json.load(f)

print(f"Results loaded from {export_path}")

# Extract features for each report
feature_matrix = []
for r in results:
    text = r["data"]["RESULT"]
    features = extract_features(text)
    feature_matrix.append(features)

# Convert to numpy array and normalize
X = np.array(feature_matrix)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Function to determine optimal number of clusters
def find_optimal_k(X_scaled, min_clusters=6, max_clusters=10):
    inertias = []
    silhouette_scores = []

    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.3f}, Inertia = {kmeans.inertia_:.3f}")

        # Calculate improvement if we have at least two scores
        if len(silhouette_scores) > 1:
            improvement = silhouette_scores[-1] - silhouette_scores[-2]
            print(f"Improvement: {improvement:.3f}")

    # Plot analysis curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    plt.subplot(1, 2, 2)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.tight_layout()
    plt.savefig("cluster_analysis.png")
    plt.close()

    # Calculate score improvements
    improvements = [
        silhouette_scores[i + 1] - silhouette_scores[i]
        for i in range(len(silhouette_scores) - 1)
    ]

    # Find where improvements become minor (less than 0.04)
    minor_improvement_threshold = 0.04
    for i, imp in enumerate(improvements):
        if imp < minor_improvement_threshold:  # found minor improvement
            optimal_k = i + min_clusters  # add min_clusters because we started from there
            break
    else:
        # If all improvements are significant, use the last value
        optimal_k = max_clusters

    print(f"\nOptimal number of clusters (k={optimal_k})")
    print("Improvements between consecutive k values:")
    for i, imp in enumerate(improvements):
        is_chosen = (i + 6) == optimal_k - 1  # Mark the k where we stopped
        print(f"K={i + 7}: {imp:.3f}" + (" *" if is_chosen else ""))

    return optimal_k


def calculate_text_overlap(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def select_diverse_samples(cluster_data, n_samples=10):
    """Select samples with minimal text overlap"""
    if len(cluster_data) <= n_samples:
        return cluster_data

    selected = []
    remaining = cluster_data.copy()

    # Select first sample (can be random or based on length)
    selected.append(remaining.pop(0))

    while len(selected) < n_samples and remaining:
        min_overlap = float("inf")
        next_sample = None
        next_index = None

        # For each remaining report
        for i, candidate in enumerate(remaining):
            # Calculate max overlap with already selected reports
            max_overlap = max(
                calculate_text_overlap(
                    candidate["data"]["RESULT"], selected_doc["data"]["RESULT"]
                )
                for selected_doc in selected
            )

            if max_overlap < min_overlap:
                min_overlap = max_overlap
                next_sample = candidate
                next_index = i

        if next_sample:
            selected.append(next_sample)
            remaining.pop(next_index)
        else:
            break

    return selected


# Find optimal k and apply clustering
optimal_k = find_optimal_k(X_scaled)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Group results by cluster and select diverse samples
grouped_results = {i: [] for i in range(optimal_k)}
for idx, label in enumerate(cluster_labels):
    grouped_results[label].append(results[idx])
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Group results by cluster
grouped_results = {i: [] for i in range(optimal_k)}
for idx, label in enumerate(cluster_labels):
    grouped_results[label].append(results[idx])

# Process each cluster
features_names = list(FEATURE_PATTERNS.keys())
for cluster_id, cluster_data in grouped_results.items():
    print(f"\nCluster {cluster_id}: {len(cluster_data)} reports")

    # Get cluster center features
    cluster_center = kmeans.cluster_centers_[cluster_id]
    cluster_center = scaler.inverse_transform([cluster_center])[0]

    # Select diverse samples
    diverse_samples = select_diverse_samples(cluster_data, n_samples=10)

    # Create cluster summary
    cluster_info = {
        "size": len(cluster_data),
        "characteristic_features": {
            features_names[i]: cluster_center[i] for i in range(len(features_names))
        },
        "selected_samples": diverse_samples,
        "remaining_reports": [r for r in cluster_data if r not in diverse_samples],
    }

    # Save to file
    # output_path = f"artifacts/results/cluster_{cluster_id}.json"
    # with open(output_path, "w") as f:
    #     json.dump(cluster_info, f, indent=2)

    # Print cluster information
    print(f"Selected {len(diverse_samples)} diverse samples")

    # Print top features
    top_features = sorted(
        zip(features_names, cluster_center), key=lambda x: abs(x[1]), reverse=True
    )[:3]
    print(f"Top features for cluster {cluster_id}:")
    for feature, value in top_features:
        print(f"  - {feature}: {value:.3f}")

    # Print average overlap within selected samples
    if len(diverse_samples) > 1:
        overlaps = []
        for i in range(len(diverse_samples)):
            for j in range(i + 1, len(diverse_samples)):
                overlap = calculate_text_overlap(
                    diverse_samples[i]["data"]["RESULT"],
                    diverse_samples[j]["data"]["RESULT"],
                )
                overlaps.append(overlap)
        avg_overlap = sum(overlaps) / len(overlaps)
        print(f"Average text overlap between selected samples: {avg_overlap:.3f}")

        # Add cluster assignment and top features to each sample
        for sample in diverse_samples:
            sample["cluster_id"] = cluster_id
            sample["cluster_features"] = {
                feature: value for feature, value in top_features
            }

# Combine all diverse samples into a final export
all_diverse_samples = []
for cluster_id, cluster_data in grouped_results.items():
    samples = select_diverse_samples(cluster_data, n_samples=10)
    for sample in samples:
        sample["cluster_id"] = cluster_id
        all_diverse_samples.append(sample)

# Create final export with metadata
final_export = {
    "total_samples": len(all_diverse_samples),
    "clusters": optimal_k,
    "samples_per_cluster": 10,
    "feature_patterns": FEATURE_PATTERNS,
    "samples": all_diverse_samples,
}

# Export detailed version (with clustering info)
detailed_export_path = "artifacts/results/diverse_samples_detailed.json"
with open(detailed_export_path, "w") as f:
    json.dump(final_export, f, indent=2)

# Create clean version (just the samples, no clustering info)
clean_samples = []
for sample in all_diverse_samples:
    clean_sample = {"id": sample["id"], "data": sample["data"]}
    if "predictions" in sample:
        clean_sample["predictions"] = sample["predictions"]
    clean_samples.append(clean_sample)

clean_export = clean_samples

# Export clean version
clean_export_path = "artifacts/results/diverse_samples.json"
with open(clean_export_path, "w") as f:
    json.dump(clean_export, f, indent=2)

print(f"\nExported {len(all_diverse_samples)} diverse samples to:")
print(f"- Detailed version: {detailed_export_path}")
print(f"- Clean version: {clean_export_path}")
