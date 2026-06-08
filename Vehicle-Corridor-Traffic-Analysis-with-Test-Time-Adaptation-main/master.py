"""
Master script to run complete Phase 1
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script"""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_path}")
        return False
    
    return True

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check if dataset is prepared
    if not Path('data/processed/yolo/visdrone3.yaml').exists():
        print("\n❌ Dataset not prepared!")
        print("Run: python scripts/prepare_datasets_v2.py")
        return False
    
    print("✅ Dataset ready")
    return True

if __name__ == '__main__':
    print("="*70)
    print("PHASE 1 - COMPLETE PIPELINE")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        exit(1)
    
    # Run all scripts in sequence
    scripts = [
        ('scripts/train_phase1.py', 'Training Detector'),
        ('scripts/test_detection.py', 'Testing Detection'),
        ('scripts/evaluate_conditions.py', 'Evaluating Conditions'),
        ('scripts/test_pipeline.py', 'Testing Pipeline'),
    ]
    
    for script, desc in scripts:
        if not run_script(script, desc):
            print(f"\n❌ Pipeline stopped at: {desc}")
            exit(1)
    
    print("\n" + "="*70)
    print("🎉 PHASE 1 COMPLETE!")
    print("="*70)
    print("\nAll outputs saved to outputs/ directory")
    print("\nYou now have:")
    print("  ✓ Trained baseline model")
    print("  ✓ Performance metrics")
    print("  ✓ Domain shift analysis")
    print("  ✓ Test videos")
    print("\n Ready for Phase 2: Test-Time Adaptation!")