"""One-off accuracy validation against 10 labeled ISIC images. Not part of the app."""

from image_processor import MelanomaDetector

RISK_THRESHOLD = 50.0  # predict "malignant" if risk_score >= this

TEST_SET = [
    ("../../Images/Benign/ISIC_0000005.jpg", "benign"),
    ("../../Images/Benign/ISIC_0000006.jpg", "benign"),
    ("../../Images/Benign/ISIC_0000007.jpg", "benign"),
    ("../../Images/Benign/ISIC_0000009.jpg", "benign"),
    ("../../Images/Benign/ISIC_0000010.jpg", "benign"),
    ("../../Images/Malignant/ISIC_0000002.jpg", "malignant"),
    ("../../Images/Malignant/ISIC_0000004.jpg", "malignant"),
    ("../../Images/Malignant/ISIC_0000030.jpg", "malignant"),
    ("../../Images/Malignant/ISIC_0000031.jpg", "malignant"),
    ("../../Images/Malignant/ISIC_0000035.jpg", "malignant"),
]


def run(detector: MelanomaDetector):
    correct = 0
    rows = []
    for path, true_label in TEST_SET:
        result = detector.process_image(path)
        risk_score = result["risk_score"]
        predicted_label = "malignant" if risk_score >= RISK_THRESHOLD else "benign"
        is_correct = predicted_label == true_label
        correct += is_correct
        rows.append((path.split("/")[-1], true_label, predicted_label, risk_score, is_correct))

    print(f"{'image':<20} {'true':<10} {'predicted':<10} {'risk_score':<10} correct")
    for name, true_label, predicted_label, risk_score, is_correct in rows:
        mark = "yes" if is_correct else "NO"
        print(f"{name:<20} {true_label:<10} {predicted_label:<10} {risk_score:<10.1f} {mark}")

    print(f"\nAccuracy: {correct}/{len(TEST_SET)} correct ({correct / len(TEST_SET) * 100:.0f}%)")
    return correct, len(TEST_SET)


if __name__ == "__main__":
    run(MelanomaDetector())
