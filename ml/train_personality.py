"""
ml/train_personality.py — Train the Personality Prediction Model
Run: python -m ml.train_personality
"""

import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.personality_model import PersonalityModel
from ml.training_data import get_training_data


def main():
    print("=" * 50)
    print("  NOVA Personality Model Trainer")
    print("=" * 50)

    # Load data
    X, y = get_training_data()
    print(f"\n📊 Training data: {len(X)} samples")
    print(f"   Classes: {sorted(set(y))}")
    for label in sorted(set(y)):
        count = y.count(label)
        print(f"   • {label}: {count} samples")

    # Train
    print("\n🔧 Training model...")
    model = PersonalityModel()
    model.train(X, y)
    print("   ✓ Training complete")

    # Save
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "personality_model.pkl")
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Quick validation
    print("\n🧪 Validation (sample predictions):")
    test_cases = [
        "explain this step by step please",
        "bro just tell me",
        "give me deep technical analysis",
        "motivate me to keep going",
        "what is the weather today",
        "help me understand machine learning",
        "yo what's good",
        "provide comprehensive architecture review",
    ]

    for text in test_cases:
        label, confidence = model.predict(text)
        bar = "█" * int(confidence * 20)
        print(f"   [{label:8s}] {confidence:.2f} {bar} │ \"{text}\"")

    # Top-N demo
    print("\n🏆 Top-3 predictions for 'explain the algorithm step by step':")
    top = model.predict_top_n("explain the algorithm step by step", n=3)
    for label, conf in top:
        print(f"   {label:8s}: {conf:.3f}")

    print("\n✅ Done! Model is ready for use.")


if __name__ == "__main__":
    main()
