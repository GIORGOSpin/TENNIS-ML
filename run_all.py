import subprocess
import sys
import time

# Βοηθητική συνάρτηση για να τρέχουμε κάθε script
def run_step(step_name, filename):
    print("\n" + "="*60)
    print(f"➡️  Τρέχει: {step_name}  ({filename}.py)")
    print("="*60)

    start = time.time()
    result = subprocess.run([sys.executable, f"{filename}.py"])

    end = time.time()
    print(f"⏳ Χρόνος εκτέλεσης: {end - start:.2f} sec")

    if result.returncode != 0:
        print(f"❌ Σφάλμα στο {step_name}! Σταματά η διαδικασία.")
        sys.exit(1)

    print(f"✅ Ολοκληρώθηκε: {step_name}")


if __name__ == "__main__":

    print("\n🚀 Ξεκινάει η πλήρης διαδικασία pipeline για ATP matches...\n")

    run_step("1. Load & Clean Data", "load_clean_data")
    run_step("2. Feature Engineering (Two-Vector)", "feature_engineering")
    run_step("3. Feature Selection", "feature_selection")
    run_step("4. Train Model (LogReg)", "train_model")
    run_step("5. Evaluate Model", "evaluate_model")

    print("\n🎉 ΟΛΟΚΛΗΡΟ ΤΟ PIPELINE ΕΚΤΕΛΕΣΤΗΚΕ ΕΠΙΤΥΧΩΣ!")
